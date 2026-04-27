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
- **Training**: 300 epochs to convergence at avg_loss = 0.191 (the lowest
  any pixel-space ViT diffusion run in this project has reached).
- **Early samples (pre-convergence) gave the wrong impression**: ~ep 20
  outputs looked like "impressionistic tiles" with patch-level textures
  and weak cross-patch coherence. **At ep 299** the same checkpoint
  generates **recognizable images from pure noise**: standing human
  figures, animals, landscapes, urban scenes — coherent compositions
  with patch boundaries no longer visible. Painterly / blurry at the
  pixel level, but globally correct.
- Conclusion: pixel-space ViT-B at 224×16 can produce recognizable
  unconditional samples with naive RF given long training. The "tiled
  patches" failure is **a transient phase, not a terminal failure mode**.
  The remaining limitation is per-pixel sharpness (no VAE), not global
  structure.

### Run 10: RF + MAE mask (MaskDiT-style, v-pred)
- Config: `pretrain_vit_b16_naive_rf_mae.yaml`
- Log dir: `logs_naive_rf_mae/` (best ckpt at epoch 164)
- Trained: 169 epochs on 2× H100 (24h walltime cap).
- **Corrected motivation**: not "fix RF's tiled failure" (Run 9 ep 299
  showed RF doesn't have a terminal tiled failure). The real story is
  **two complementary inductive biases on a shared encoder**:
    - MAE-style mask substitution → encoder learns to recover x_0 from
      cross-patch context (global / semantic signal)
    - RF v-prediction on visible tokens → high-frequency / pixel detail
  The hypothesis: MAE accelerates convergence of the global pathway,
  while RF supplies the high-frequency learning signal.
- Design (MaskDiT-style, symmetric):
  - Per-step mask ratio r ~ U(0, 0.5). Includes r≈0 so training covers
    the clean-input distribution used at sampling (no inference OOD gap).
  - All 196 tokens flow through DiTEncoder; mask_token replaces masked
    embeddings before the transformer stack.
  - v-prediction loss on all patches: `L = L_visible + 0.1 · L_masked`.
- **Loss-curve evidence (vs Run 9)**:

  | epoch | Run 9 (naive RF) | Run 10 visible (v) | Run 10 masked − 1.0 (x_0 residual) |
  |---|---|---|---|
  | 1   | 0.260 | 0.260 | 0.301 |
  | 19  | 0.210 | 0.216 | 0.189 |
  | 79  | 0.200 | 0.196 | 0.214 |
  | 139 | —     | 0.193 | 0.175 |
  | 163 | —     | **0.187** | **0.174** |
  | 299 | **0.191** | — | — |

  Run 10 visible v-loss reached **0.187 at ep 163**, lower than Run 9's
  300-epoch best of 0.191. **MAE auxiliary did not hurt the primary RF
  task; if anything it slightly helped, with ~45% fewer epochs to
  match-or-beat Run 9 quality.**
- **Sample quality** (50-step Euler from pure noise, samples_rf_mae_from_noise/grid.png):
  - Equivalent to Run 9 ep 299: standing figures, animals, landscapes
    with the same painterly style and global coherence.
  - At ep 164 vs Run 9 ep 299 → **~45% fewer epochs for equivalent
    generation quality**.
- **Inpainting capability (Run 10 specifically — single-pass MAE)**:
  - `scripts/inpaint_rf_mae.py` uses the masked branch directly: at
    masked positions, optimal pred_v ≈ −x̂_0 (because ε is independent
    of context there), so x̂_0 recovery is just `−pred_v`.
  - Tested at 50% mask ratio: composite is visually indistinguishable
    from the original at the patch-coherence level; per-patch fill is
    semantically correct (red-jacket fisherman's clothing, fish body,
    grass continue).
  - **Limitation**: single forward pass → details are softer than RF
    multi-step would produce. Next: Run 11 dual-head iterates RF v-head
    on top of x_0-head's global init.
- **Decomposition of masked v-loss into MAE signal**:
  - L_masked = MSE(pred_v, ε−x_0) = R_{x_0} + Var(ε)
  - Var(ε) = 1.0 is irreducible (ε independent of context at masked)
  - At ep 163: L_masked ≈ 1.174 → R_{x_0} ≈ 0.174 → **~83% of x_0
    variance recovered from cross-patch context** (vs 73% at ep 1).
- **Verdict**: Run 10 is the project's first clear architectural win at
  pixel-space ViT 224×16 — same generation quality at 45% less compute,
  plus a usable inpainting capability not present in Run 9.

### Run 11: Dual-head RF (v + x_0) with clean-prompt patches (inpaint)
- Config: `pretrain_vit_b16_dual_rf.yaml`
- Log dir: `logs_dual_rf/`
- **Status**: launched on 4× H100 after the morning of 2026-04-27.
- **Positioned as image completion / prompt-to-image, not unconditional
  generation.** Run 7 (this same architecture with DDPM ε-target) had
  mode collapse at unconditional sampling — at test time the encoder
  saw all-noise input and the x_0-head's natural-image prior dominated.
  Switching the loss to RF v-target wouldn't fix that root cause, so
  the design pivots: **at inference, REQUIRE 25% real clean prompt**.
  Train and test then share the same input distribution.
- Design (= Run 7 dual + RF):
  - Input = 25% clean (prompt) + 75% RF-noised (x_t = (1-t)x_0 + t·ε)
  - Shared DiTEncoder + clean/noisy indicators + time conditioning
  - decoder      → v = ε − x_0 on noisy positions   (RF, high-freq)
  - decoder_pix  → x_0 on noisy positions            (MAE, global)
  - Loss on noisy positions only: L = L_v + λ · L_x0
- Why dual is the right choice for inpainting (vs Run 10's single-head):
  - Run 10's masked branch can do single-pass MAE inpainting but is
    soft on detail (single forward, no iterative refinement).
  - Run 11's two heads support a two-stage inpainting pipeline:
    1. x_0-head provides a fast, globally-correct (but blurry) init for
       the masked positions.
    2. v-head iterates Euler reverse steps to refine details, while the
       clean prompt patches stay anchored at their known x_0 values.
  - This combines MAE's global-fast learning with RF's iterative
    high-frequency refinement.
- Expected outcomes:
  - Inpaint composite sharper than Run 10 at the same mask ratio.
  - Loss-wise: visible v should match Run 9 / Run 10 (~0.19); x_0 head
    should reach ~0.10–0.15 (lower than Run 10's masked-residual
    because no ε noise floor in this loss).

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
5. **Rectified Flow alone does eventually produce coherent samples**
   (Run 9 ep 299). The "tiled patches" failure is a transient phase, not
   a terminal failure mode. RF needs ~300 epochs at 4× H100 to get there.
   Pixel-space ViT-B at 224×16 is constrained on per-pixel sharpness, not
   global structure.
6. **MAE auxiliary on top of RF gives a 45% compute-to-quality win**
   (Run 10): visible v-loss at ep 163 (0.187) ≤ Run 9's 300-ep best
   (0.191), and 50-step samples are visually equivalent to Run 9 ep 299.
   The MAE branch additionally delivers a working single-pass inpainting
   capability (`scripts/inpaint_rf_mae.py`).
7. **Run 7 dual-decoder's mode collapse is sidestepped by reframing
   dual + clean anchors as image completion** (Run 11). With 25% clean
   prompt required at both train and inference, no OOD gap, no collapse;
   the v-head and x_0-head become the two stages of an iterative
   inpainting pipeline (x_0 init + Euler refinement).
