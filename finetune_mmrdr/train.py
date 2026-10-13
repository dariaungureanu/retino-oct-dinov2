"""
MMRDR-OCT Fine-Tuning - 3-class DME severity grading.
Modified for Statistical Robustness: Runs 5 random seeds and reports Mean +/- SD.

Single-task version of the OCTDL pipeline.
Metrics: Accuracy, Balanced Acc, Macro-F1, AUC-ROC, Cohen's Kappa
(matches the paper's evaluation for direct comparison).

Usage:
    # Domain-adapted backbone
    python finetune_mmrdr/train.py \
        --data_path /home/student/Ungureanu_Daria/MMRDR-OCT \
        --csv //home/student/Ungureanu_Daria/MMRDR-OCT/OCT.csv \
        --checkpoint /home/student/Ungureanu_Daria/antrenare_oct_v2/model_final.rank_0.pth \
        --save_dir saved_models/mmrdr_domain_adapted \
        --run_name "mmrdr_domain_adapted"

    # ImageNet baseline (no --checkpoint)
    python finetune_mmrdr/train.py \
        --data_path /home/student/Ungureanu_Daria/MMRDR-OCT \
        --csv /home/student/Ungureanu_Daria/MMRDR-OCT/OCT.csv \
        --save_dir saved_models/mmrdr_imagenet_aug \
        --run_name "mmrdr_imagenet_aug"
"""

import argparse
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    cohen_kappa_score, roc_auc_score, confusion_matrix
)
from sklearn.preprocessing import label_binarize
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import (
    MMRDRDataset, load_mmrdr_splits, compute_class_weights,
    get_train_transform, get_eval_transform, CLASS_NAMES
)
from model import MMRDRModel, load_backbone

ARCH = "dinov2_vits14"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
LR_BACKBONE = 1e-5
LR_HEADS = 5e-4
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 3
GRAD_CLIP = 1.0
UNFREEZE_LAST_N = 2
HEAD_HIDDEN = 256
HEAD_DROPOUT = 0.3
PATIENCE = 8
NUM_WORKERS = 4


def compute_all_metrics(logits_or_probs, labels, num_classes):
    """Compute accuracy, balanced_accuracy, macro_f1, auc_roc, kappa (paper's metric set)."""
    if isinstance(logits_or_probs, torch.Tensor):
        probs = torch.softmax(logits_or_probs, dim=1).cpu().numpy()
        preds = torch.argmax(logits_or_probs, dim=1).cpu().numpy()
        labels = labels.cpu().numpy()
    else:
        probs = logits_or_probs
        preds = np.argmax(probs, axis=1)

    acc = accuracy_score(labels, preds) * 100
    bal_acc = balanced_accuracy_score(labels, preds) * 100
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    kappa = cohen_kappa_score(labels, preds)

    try:
        labels_bin = label_binarize(labels, classes=list(range(num_classes)))
        auc_roc = roc_auc_score(labels_bin, probs, multi_class="ovr", average="macro")
    except ValueError:
        auc_roc = 0.0

    return {
        "acc": acc,
        "bal_acc": bal_acc,
        "macro_f1": macro_f1,
        "auc_roc": auc_roc,
        "kappa": kappa,
    }


def run_epoch(model, loader, criterion, device, num_classes,
              optimizer=None, scheduler=None, grad_clip=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_logits, all_labels = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total_loss += loss.item()
            all_logits.append(logits.detach())
            all_labels.append(labels.detach())

    if is_train and scheduler is not None:
        scheduler.step()

    cat_logits = torch.cat(all_logits)
    cat_labels = torch.cat(all_labels)
    avg_loss = total_loss / len(loader)
    metrics = compute_all_metrics(cat_logits, cat_labels, num_classes)

    return avg_loss, metrics

def evaluate_test(model, loader, criterion, device, num_classes, out_dir, seed):
    model.eval()
    all_logits, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"Test Eval (Seed {seed})", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            all_logits.append(logits)
            all_labels.append(labels)

    cat_logits = torch.cat(all_logits)
    cat_labels = torch.cat(all_labels)
    metrics = compute_all_metrics(cat_logits, cat_labels, num_classes)

    preds = torch.argmax(cat_logits, dim=1).cpu().numpy()
    labels_np = cat_labels.cpu().numpy()
    class_names = [CLASS_NAMES[i] for i in range(num_classes)]

    cm = confusion_matrix(labels_np, preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted", fontweight="bold")
    ax.set_ylabel("True", fontweight="bold")
    ax.set_title(f"MMRDR-OCT DME Confusion Matrix (Seed {seed})", fontweight="bold")
    plt.tight_layout()
    cm_path = os.path.join(out_dir, f"confusion_matrix_seed{seed}.png")
    fig.savefig(cm_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return metrics

def main():
    parser = argparse.ArgumentParser(description="MMRDR-OCT DME Fine-Tuning")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="saved_models/mmrdr")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr_backbone", type=float, default=LR_BACKBONE)
    parser.add_argument("--lr_heads", type=float, default=LR_HEADS)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--warmup_epochs", type=int, default=WARMUP_EPOCHS)
    parser.add_argument("--grad_clip", type=float, default=GRAD_CLIP)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--arch", type=str, default=ARCH)
    parser.add_argument("--img_size", type=int, default=IMG_SIZE)
    parser.add_argument("--unfreeze_last_n", type=int, default=UNFREEZE_LAST_N)
    parser.add_argument("--head_hidden", type=int, default=HEAD_HIDDEN)
    parser.add_argument("--head_dropout", type=float, default=HEAD_DROPOUT)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    train_df, val_df, test_df, num_classes = load_mmrdr_splits(
        args.csv, args.data_path, val_size=0.1,
    )
    weights = compute_class_weights(train_df, num_classes).to(device)

    seeds = [42, 123, 456, 789, 999]
    metrics_history = {"acc": [], "bal_acc": [], "macro_f1": [], "auc_roc": [], "kappa": []}

    for seed in seeds:
        print(f"\n{'='*50}\nSTARTING RUN WITH SEED {seed}\n{'='*50}")
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)

        train_ds = MMRDRDataset(train_df, args.data_path, get_train_transform(args.img_size))
        val_ds = MMRDRDataset(val_df, args.data_path, get_eval_transform(args.img_size))
        test_ds = MMRDRDataset(test_df, args.data_path, get_eval_transform(args.img_size))

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True)

        backbone = load_backbone(args.arch, args.checkpoint, device)
        model = MMRDRModel(
            backbone=backbone, num_classes=num_classes,
            freeze_backbone=(args.unfreeze_last_n < 12),
            unfreeze_last_n=args.unfreeze_last_n,
            head_hidden=args.head_hidden, head_dropout=args.head_dropout,
        ).to(device)

        param_groups = model.get_param_groups(args.lr_backbone, args.lr_heads, args.weight_decay)
        optimizer = optim.AdamW(param_groups)
        warmup = LinearLR(optimizer, start_factor=0.01, total_iters=args.warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=1e-7)
        scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[args.warmup_epochs])
        criterion = nn.CrossEntropyLoss(weight=weights)

        run_name = f"{args.run_name or 'mmrdr'}_seed{seed}"
        wandb.init(project="MMRDR-OCT-FineTune", name=run_name, config=vars(args), reinit=True)

        best_val_f1 = 0.0
        patience_counter = 0
        best_epoch = 0
        save_path = os.path.join(args.save_dir, f"best_model_seed{seed}.pth")

        for epoch in range(1, args.epochs + 1):
            t_loss, t_met = run_epoch(model, train_loader, criterion, device, num_classes,
                                      optimizer=optimizer, scheduler=scheduler, grad_clip=args.grad_clip)
            v_loss, v_met = run_epoch(model, val_loader, criterion, device, num_classes)

            wandb.log({
                "epoch": epoch, "train/loss": t_loss, "train/f1": t_met["macro_f1"],
                "val/loss": v_loss, "val/f1": v_met["macro_f1"],
            })

            if v_met["macro_f1"] > best_val_f1:
                best_val_f1 = v_met["macro_f1"]
                best_epoch = epoch
                patience_counter = 0
                torch.save({"model_state_dict": model.state_dict()}, save_path)
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stop at epoch {epoch}")
                    break

        print(f"Testing best model (epoch {best_epoch}) for seed {seed}...")
        best_ckpt = torch.load(save_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])
        test_met = evaluate_test(model, test_loader, criterion, device, num_classes, args.save_dir, seed=seed)

        for k in metrics_history.keys():
            metrics_history[k].append(test_met[k])
        wandb.finish()

    print("\n" + "="*50)
    print("FINAL ROBUSTNESS RESULTS (5 Seeds)")
    print("="*50)
    final_summary = {}
    for k, v in metrics_history.items():
        mean_val = np.mean(v)
        std_val = np.std(v)
        ci_95 = 1.96 * (std_val / np.sqrt(len(seeds)))
        final_summary[k] = {"mean": mean_val, "std": std_val, "ci_95": ci_95}
        
        if k in ["acc", "bal_acc"]:
            print(f"{k}: {mean_val:.2f}% ± {std_val:.2f}% (95% CI: ±{ci_95:.2f}%)")
        else:
            print(f"{k}: {mean_val:.4f} ± {std_val:.4f} (95% CI: ±{ci_95:.4f})")

    json_path = os.path.join(args.save_dir, "robustness_results.json")
    with open(json_path, "w") as f:
        json.dump({"seeds": seeds, "raw": metrics_history, "summary": final_summary}, f, indent=2)
    print(f"Results saved to {json_path}")

if __name__ == "__main__":
    main()
