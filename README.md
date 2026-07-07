# Domain-Adaptive Continual Pre-training of DINOv2 for Retinal OCT

This repository accompanies the paper and provides the code for the downstream
evaluation of a domain-adapted ViT-S/14 backbone on retinal optical coherence
tomography (OCT): supervised fine-tuning on four labeled tasks, frozen-feature
evaluation of the backbone, and the explainability analyses. It reproduces every
downstream result and figure reported in the paper given the trained
checkpoints and the public datasets.

The self-supervised pre-training stage that produces the domain-adapted backbone
was carried out with the official Meta DINOv2 reference implementation and is not
redistributed here; the resulting backbone is provided as a checkpoint (see
[Checkpoints](#checkpoints)).

## Repository layout

```
finetune_octdl/      OCTDL multi-task disease + condition classification
finetune_mmrdr/      MMRDR-OCT three-class DME severity grading
finetune_corina/     Suciu et al. multi-label DME-biomarker detection
finetune_oct5k/      OCT5k multi-label biomarker detection
analyse_pretrain/    Frozen-feature probes on the backbone (kNN, linear probe,
                     UMAP, patch-PCA, CLS-patch cosine similarity)
PREPROCESS_DATA/     OCTDL cleaning (adaptive flood-fill artifact removal)
requirements.txt     Python dependencies
```

Each `finetune_*` package follows the same structure:

- `dataset.py` — data loading, splits, and label construction
- `model.py` — the ViT-S/14 backbone with the task-specific head(s)
- `train.py` — training entry point (domain-adapted or ImageNet baseline)
- `analyse_*.py` — evaluation, confusion matrices, Grad-CAM, and t-SNE

## Environment

```
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The results in the paper were produced with PyTorch on a single NVIDIA RTX 3090
(24 GB). A fixed random seed of 42 controls data splits, head initialization,
and dataloader shuffling, with PyTorch deterministic mode enabled.

The backbone is instantiated through `torch.hub` from the public
`facebookresearch/dinov2` `dinov2_vits14` definition; domain-adapted weights are
then loaded from the checkpoint. Training runs log to Weights & Biases; pass the
run/project arguments exposed by each `train.py` or disable logging in your
environment as preferred.

## Data availability

The four labeled datasets are obtained from their original public release pages
and are not redistributed here:

- **OCTDL** — Kulyabin et al.
- **MMRDR-OCT** (OCT subset) — Tang et al.
- **Suciu et al.** DME-biomarker OCT dataset
- **OCT5k** — Arikan et al.

The institutional pre-training collection from the Cluj-Napoca Emergency County
Hospital is held private and is not part of this repository or the downstream
evaluation. Dataset splits are patient-level (except MMRDR-OCT, which inherits
the released image-level partition) and are reconstructed deterministically by
the `dataset.py` in each package.

## Checkpoints

The domain-adapted ViT-S/14 backbone and the per-task fine-tuned models are
distributed separately through a shared drive; the link is provided below.

**[checkpoints.zip](https://drive.google.com/file/d/1vgN-RgZO_Ys_jNBQWXjDcO2Yizsv9Uho/view?usp=sharing)** (Google Drive)

Expected files:

- `model_final.rank_0.pth` — the domain-adapted (DA) backbone
- `<task>_da.pth`, `<task>_in.pth` — per-task fine-tuned models for the
  domain-adapted and ImageNet-baseline backbones (OCTDL, MMRDR, Suciu, OCT5k)

## Reproducing the downstream results

Paths shown below are examples; each script prints its full argument list with
`--help`. Every task trains two models under identical seeding and data
ordering: the domain-adapted backbone (pass `--checkpoint`) and the ImageNet
baseline (omit `--checkpoint`), so each comparison isolates the contribution of
domain-adaptive pre-training.

### 1. Preprocess OCTDL

```
python PREPROCESS_DATA/preprocess_data.py --help
```

Removes the black vitreous band and burned-in scanner annotations by adaptive
flood-fill and writes the cleaned images plus a metadata CSV.

### 2. Fine-tune each task

```
# OCTDL — multi-task disease + condition (domain-adapted)
python finetune_octdl/train.py --data_path <OCTDL_Cleaned> \
    --checkpoint <model_final.rank_0.pth> --unfreeze_last_n 2

# MMRDR-OCT — DME severity grading
python finetune_mmrdr/train.py --data_path <MMRDR-OCT> --csv <OCT.csv> \
    --checkpoint <model_final.rank_0.pth>

# Suciu et al. — multi-label DME biomarkers
python finetune_corina/train.py --data_path <corina_dataset> --csv <corina_metadata.csv> \
    --checkpoint <model_final.rank_0.pth>

# OCT5k — multi-label biomarkers (build the label CSV first)
python finetune_oct5k/build_csv.py --data_path <oct5k> --out_csv <oct5k_metadata.csv>
python finetune_oct5k/train.py --data_path <oct5k> --csv <oct5k_metadata.csv> \
    --checkpoint <model_final.rank_0.pth>
```

Omit `--checkpoint` in any of the above to train the ImageNet baseline.

### 3. Evaluation, explainability, and additional analyses

```
# Metrics, confusion matrices, Grad-CAM, t-SNE (per task)
python finetune_octdl/analyse_explainability.py --help
python finetune_mmrdr/analyse_mmrdr.py --help
python finetune_corina/analyse_corina.py --help
python finetune_oct5k/analyse_oct5k.py --help

# OCTDL-specific analyses
python finetune_octdl/analyse_confusion.py --help
python finetune_octdl/analyse_disease_vs_condition.py --help
python finetune_octdl/analyse_patient_level.py --help    # majority-vote / avg-probability
python finetune_octdl/data_efficiency.py --help          # 33% / 66% / 100% patient subsets
```

For OCT5k the explainability analysis is quantitative: per-biomarker Grad-CAM
heatmaps are binarized and scored by intersection-over-union against the
ground-truth bounding boxes.

### 4. Frozen-feature evaluation of the backbone

Applied identically to the domain-adapted backbone and the public LVD-142M
DINOv2 weights (omit `--checkpoint` for the baseline):

```
python analyse_pretrain/method_frozen_eval.py --help    # kNN + linear probe
python analyse_pretrain/method_umap.py --help           # UMAP of CLS tokens
python analyse_pretrain/method_patch_pca.py --help      # patch-PCA overlays
python analyse_pretrain/method_cls_similarity.py --help # CLS-patch cosine maps
```

## Citation

If you use this code, please cite the accompanying paper. A BibTeX entry will be
added here upon publication.
