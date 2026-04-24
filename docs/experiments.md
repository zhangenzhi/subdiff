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
  architecture** — initially read as positive transfer from pix head to
  ε head; subsequent sampling revealed this was **mode collapse** (see
  stability.md § "Failure mode 4").

### Run 8: patch_size=2 at 32×32 (patch-size ablation, DDPM)
- Config: `pretrain_vit_b16_naive_ddpm_p2_32.yaml`
- Log dir: `logs_naive_ddpm_p2_32/`
- Goal: decouple patch-size from model-size to test whether the
  "bag-of-patches / tiled mean" failure mode at 224×16 is caused by the
  Linear head acting per-16×16-patch.
- Setup: dataloader resizes ImageNet to 32×32, ViT with patch_size=2 →
  also 16×16 = 256 tokens, same ViT-B depth/width. Everything else
  identical to Run 5 (DiTEncoder + QK-Norm + minimal head + constant lr).
- **Outcome: samples show real images** — recognizable object silhouettes
  and textures, not tiles. This is the first pixel-space ViT diffusion
  run in this project to produce coherent 32×32 samples.
- Conclusion: patch size is the core bottleneck at 224×16. A 16×16 Linear
  head cannot produce intra-patch structure; the ViT attention carries
  information, but the per-token output space is too coarse. At 2×2 the
  head has enough pixel resolution and attention does the cross-patch
  work. Latent DiT sidesteps this by having the VAE decoder provide the
  intra-patch structure.
- Implication: to get recognizable 224×224 output without a VAE we need
  EITHER (a) replace the Linear head with something that has intra-patch
  spatial structure (Conv refine, convolutional head, unpatchify + UNet
  decoder), OR (b) switch to a training objective that is more robust to
  the low-dim-head bottleneck (flow matching with x-pred in JiT; see
  Runs 9-10).

### Run 9: naive Rectified Flow at 224×16 (SD3/FLUX-style)
- Config: `pretrain_vit_b16_naive_rf.yaml`
- Log dir: `logs_naive_rf/`
- Goal: test whether replacing DDPM ε-pred with Rectified Flow v-pred
  improves per-patch generation quality at 224×16. SD3 ingredients:
  linear interpolation x_t = (1-t)x_0 + t·ε, v = ε - x_0, logit-normal t
  sampling (μ=0, σ=1), simple MSE loss.
- Backbone identical to Run 5 (DiTEncoder + QK-Norm + minimal head).
  Model routes through new `_forward_naive_rf` + `RectifiedFlow` module
  in `subdiff/diffusion.py`.
- New sampling script: `scripts/sample_flow.py` (Euler / Heun ODE
  solvers, default 50 steps).
- **Early training**: loss descends cleanly from ~1.0 toward ~0.5 in
  epoch 0 (comparable to DDPM ε-pred curve).
- **Early samples (pre-convergence)**: still "impressionistic tiles"
  — per-patch textures are reasonable but no cross-patch object
  structure. Not worse than naive DDPM, not obviously better either.
  v-prediction alone does not fix the 224×16 coherence problem.
- Diagnosis: the head is still per-token Linear with patch_dim=768.
  RF changes the *loss target* but not the architectural bottleneck.
  The per-patch independence is a **head problem**, not a loss problem.

### Run 10: RF + MAE mask (MaskDiT-style, v-pred)
- Config: `pretrain_vit_b16_naive_rf_mae.yaml`
- Log dir: `logs_naive_rf_mae/`
- Status: **implemented, not yet launched**.
- Motivation: naive RF (Run 9) produces "impressionistic tiles" because
  each patch prediction is near-independent. Replacing a fraction of
  patch tokens with a learnable `mask_token` BEFORE the encoder forces
  attention to reconstruct them from context only, which is expected to
  add cross-patch coherence at the expense of per-patch sharpness.
- Design (MaskDiT-style, symmetric):
  - Per-step mask ratio r ~ U(0, 0.5). Includes r ≈ 0 so training still
    covers the clean-input distribution used at sampling (no inference
    OOD gap).
  - All 196 tokens flow through the DiTEncoder (no asymmetric
    encoder-decoder — our minimal Linear head is per-token anyway).
  - v-prediction loss on all patches: `L = L_visible + 0.1 · L_masked`.
- Implementation: new `_forward_naive_rf_mae` method on `SubDiff`,
  gated by `diffusion.rf_mae_enabled`. Routed in `forward()` before
  the existing RF branch when flow_matching AND naive_ddpm AND
  rf_mae_enabled are all true. New parameter `rf_mask_token`
  (`(1, 1, embed_dim)`, trunc_normal init).
- Sampling at inference time: use `scripts/sample_flow.py` unchanged —
  sampling naturally operates in the r=0 regime, which matches the
  r≈0 tail of the training distribution.
- Expected outcome vs Run 9:
  - More coherent (object contours emerge as attention learns cross-
    patch priors).
  - Still blurry at pixel level (v-pred cannot produce fine detail
    without a VAE; this is the same bottleneck Run 8 diagnosed).

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
4. **Patch size is the dominant bottleneck for pixel-space ViT diffusion at
   224×16** (Run 8). At 32×32 with patch_size=2 the same architecture
   generates recognizable images; at 224×16 the per-token Linear head cannot
   express intra-patch structure and output degenerates to textured tiles.
   Latent DiT sidesteps this via the VAE decoder; for pixel-space we need
   either a structured head or a different formulation.
5. **Rectified Flow alone does not fix the coherence problem** (Run 9).
   Changing ε → v target changes the loss landscape but not the per-token
   head bottleneck. The "tiled" failure mode is architectural, not
   loss-related.
6. **Next hypothesis: RF + MAE mask** (Run 10, implemented) aims to add
   cross-patch coherence via mask-token substitution in the encoder, while
   keeping RF's stable training signal. This is the current best bet for
   improving 224×16 generation without abandoning pixel space.
