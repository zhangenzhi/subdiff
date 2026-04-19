# Experiment Log

All runs: ViT-B/16 backbone, ImageNet-1K, 4× H100, bf16 mixed precision.

## Pretraining runs

### Run 1: Original SubDiff (pixel reconstruction, curriculum)
- Config: `pretrain_vit_b16.yaml`
- Log dir: `logs/`
- Epochs: 300 (early job killed at ep 267 by walltime, resumed to 300)
- Checkpoint: `logs/checkpoints/checkpoint_final.pth` (old naming)
- Final noisy_loss: ~0.058 at ep 267 (curriculum t=[122, 612])

### Run 2: Noise prediction (predict_noise, curriculum)
- Config: `pretrain_vit_b16_eps.yaml`
- Log dir: `logs_eps/`
- Epochs: ~230 when inspected
- Behavior: loss dropped sharply to 0.37 at ep 10, rebounded to ~0.98 and
  plateaued. Attributed to curriculum-epsilon mismatch (see designs.md).

### Run 3: MAE-masking (not completed / superseded)
- Config: `pretrain_vit_b16_mae.yaml`
- Discarded before meaningful training due to design flaw: visible patches
  are all noisy, no clean anchors, at high t the encoder has no spatial signal.

### Run 4: Dual-decoder (early / superseded)
- Config: `pretrain_vit_b16_dual.yaml`
- Log dir: `logs_dual/` then `logs_dual_qknorm/`
- First attempt crashed: DDP found unused parameters. Fix: DDP with
  `find_unused_parameters=True`.
- Eventually exposed stability issues (see `stability.md`); superseded by
  Run 7 (dual_dit) after architecture cleanup.

### Run 5: naive DDPM-ViT (stability baseline for generation)
- Configs: several iterations leading to
  `pretrain_vit_b16_naive_ddpm_minimal.yaml`
- Key final configuration: DiTEncoder (per-block adaLN-Zero) + QK-Norm +
  single zero-init Linear head + constant lr 1e-4 + no weight decay.
- Log dir: `logs_naive_ddpm_minimal/`
- Status (epoch 5+): training loss 0.05-0.10, converging cleanly. Sampling
  produces patch-level structure (clear breakthrough from earlier pure-noise
  outputs).

### Run 6: SubDiff eps_qknorm with DiTEncoder (single ε head)
- Config: `pretrain_vit_b16_eps_qknorm.yaml`
- Log dir: `logs_eps_dit/`
- Same backbone as Run 5 but with SubDiff's 25% clean anchors, noise/clean
  indicator embeddings, predict_noise=True.
- Epoch 0 end: avg_loss 0.117, ε loss ≈ naive_ddpm.

### Run 7: SubDiff dual with DiTEncoder (ε + pixel heads)
- Config: `pretrain_vit_b16_dual.yaml`
- Log dir: `logs_dual_dit/`
- Same backbone as Runs 5-6 but with dual heads: ε prediction + pixel
  reconstruction, both with minimal (zero-init Linear) heads. Loss computed
  only on noisy patches.
- Epoch 0 end: avg_loss 0.527, ε loss **0.099**, pix loss 0.33.
- **ε loss ~17% lower than Run 5 (0.099 vs 0.117) with identical
  architecture** — tentative evidence of positive transfer from pix head
  to ε head (see stability.md § "Positive finding").

## Downstream finetuning

### Classification finetune

| Setup | Ep | val_top1 | val_top5 | Notes |
|---|---|---|---|---|
| Scratch (baseline) | 27 | 0.6166 | 0.8286 | ViT-B/16 from random init |
| Run 1 pretrain → finetune | 86 | **0.7265** | 0.8904 | +11% over scratch |

The +11% on top1 is the main signal that pretraining works for the
discriminative pathway. Baseline ViT-B/16 scratch on ImageNet without heavy
augmentation typically plateaus at ~61%, matching what we observe.

### Diffusion finetune (epsilon prediction)

Two variants tested:

**(a) Different patch sizes (decoder patch 8, encoder patch 16)**

Run 1 pretrained encoder vs scratch, comparing loss at matching steps:

| Step | Pretrained | Scratch |
|---|---|---|
| 100 | 0.9161 | 0.9200 |
| 500 | 0.8439 | 0.8957 |
| 1000 | 0.7473 | 0.6855 |
| 2000 | 0.6539 | 0.5702 |

Pretrained leads at early steps (up to ~500), then scratch surpasses. No net
speedup.

**(b) Same patch sizes (both 16), with decoder weight transfer**

Tested transferring the pretrained decoder's self-attention + MLP weights into
the diffusion decoder (configs: `finetune_diffusion_initdec.yaml` and
`scratch_diffusion_p16.yaml`):

| Step | Pretrained+InitDec | Scratch |
|---|---|---|
| 100 | 1.0014 | 1.0034 |
| 500 | 0.8439 | 0.8957 |
| 1000 | 0.7473 | 0.6855 |
| 2000 | 0.6539 | 0.5702 |

Essentially the same pattern. Decoder weight transfer does not help either.

**Diagnosis** (see designs.md): Run 1's pixel-reconstruction pretraining
produces a decoder that learned "predict clean pixels," not "predict epsilon."
Even with weight transfer, the parameters are poorly initialized for the new
objective. Also, most diffusion training signal comes from low t where the
noisy patch already carries most of the information — the decoder doesn't
need the encoder's conditioning, and learns to ignore it (shortcut).

## Checkpointing policy (changed mid-project)

Old policy: save `checkpoint_XXXX.pth` every `save_every` epochs. Disk usage
was 59 GB for a 300-epoch run.

New policy: keep only
- `checkpoint_latest.pth` — overwritten each save.
- `checkpoint_best.pth` — overwritten when `avg_loss` improves.

Resume reads `best_loss` from the checkpoint to continue tracking correctly.

## Takeaways from experiments so far

1. **Pretraining clearly helps classification** (Run 1 + cls finetune: +11%).
   The representation is semantic enough to benefit discriminative transfer.
2. **Pretraining does not obviously help diffusion** (Run 1 + diff finetune:
   no net speedup). Two plausible causes, likely both:
   - Target mismatch: pixel pretraining produces features misaligned with the
     epsilon-prediction downstream.
   - Decoder shortcut: at low t the diffusion decoder can ignore encoder
     conditioning entirely.
3. The dual-decoder design (Run 4) is a direct response: align one head with
   the epsilon downstream, while the pixel head continues to supply the
   semantic signal that worked for classification.
