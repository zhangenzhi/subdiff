# SubDiff: Sub-image Patch Diffusion Pretraining

Self-supervised visual pretraining that combines patch-level diffusion denoising with spatial reconstruction.

## Method

Given an image divided into patches:
- **75% of patches** are corrupted with strong diffusion noise
- **25% of patches** remain clean as spatial anchors
- All patches (noisy + clean) are fed into a ViT encoder
- A lightweight decoder reconstructs the original clean patches

**Curriculum learning** gradually transitions the task:
- **Early training**: strong noise (t ≈ T) — noisy patches ≈ pure noise, forcing global spatial reasoning (similar to MAE)
- **Late training**: weaker noise — model refines local denoising and learns fine-grained features

This unifies the benefits of MAE (global structure) and diffusion (noise-robust features) in a single pretraining stage.

## Project Structure

```
subdiff/
├── configs/
│   └── pretrain_vit_b16.yaml    # Training config
├── subdiff/
│   ├── vit.py                   # ViT encoder + decoder
│   ├── diffusion.py             # Patch-level diffusion noise
│   ├── curriculum.py            # Curriculum learning scheduler
│   ├── model.py                 # SubDiff main model
│   └── data.py                  # ImageNet data loading
├── scripts/
│   ├── pretrain.py              # Pretraining script (DDP)
│   └── linear_probe.py          # Linear probe evaluation
└── requirements.txt
```

## Usage

### Setup

```bash
pip install -r requirements.txt
```

### Pretraining

Edit `configs/pretrain_vit_b16.yaml` to set your ImageNet path.

```bash
# Single GPU
python scripts/pretrain.py --config configs/pretrain_vit_b16.yaml

# Multi-GPU (8 GPUs)
torchrun --nproc_per_node=8 scripts/pretrain.py --config configs/pretrain_vit_b16.yaml

# Resume from checkpoint
python scripts/pretrain.py --config configs/pretrain_vit_b16.yaml --resume logs/checkpoints/checkpoint_0100.pth
```

### Linear Probe Evaluation

```bash
python scripts/linear_probe.py \
    --config configs/pretrain_vit_b16.yaml \
    --checkpoint logs/checkpoints/checkpoint_final.pth
```

## Curriculum Schedule

| Training Phase | Noise Strength | Clean Ratio | Effective Task |
|---|---|---|---|
| Early (warm-up) | t ∈ [800, 1000] | 25% | ≈ MAE (global reasoning) |
| Mid | t ∈ [400, 800] | 15% | Mixed (structure + denoising) |
| Late | t ∈ [100, 600] | 5% | Fine denoising |
