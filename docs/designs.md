# Design Evolution

Each design below solves a specific limitation of its predecessor. Configs are
in `configs/`, code paths are in `subdiff/model.py`.

## 1. Original SubDiff (baseline)

**Config**: `pretrain_vit_b16.yaml` (`predict_noise: false`, no mae_masking).

**Design**:
- Patch split into `clean_ratio` clean + `(1 - clean_ratio)` noisy.
- Shared timestep t per image, sampled from curriculum-controlled range
  `[t_min, t_max]`.
- Single decoder predicts the clean pixel values.
- Loss: `L_pix(noisy) + 0.1 * L_pix(clean)`.
- Curriculum: `[t_min, t_max]` decays from `[800, 1000]` to `[100, 600]`;
  `clean_ratio` decays from 0.25 to 0.05.

**Result**: classification finetune top1 72.65 vs scratch 61.5 (+11). Strong
discriminative transfer.

**Limitation discovered**:
- Curriculum late stages are "too easy": low t means the input is already
  nearly clean, so single-step pixel reconstruction approaches identity. The
  model stops being forced to do global reasoning.
- Pretraining target (clean pixel) does not match downstream diffusion target
  (epsilon). Decoder weights do not transfer.
- Diffusion downstream finetuning with pretrained encoder is no faster than
  scratch (pretrained briefly leads at early steps, then scratch surpasses).

## 2. Noise prediction (predict_noise)

**Config**: `pretrain_vit_b16_eps.yaml` (`predict_noise: true`).

**Change**: decoder target switches from clean pixel to noise epsilon. Same
structure otherwise (25% clean anchors, curriculum-controlled t range).

**Rationale**: align pretraining target with downstream diffusion so the
decoder weights can be reused.

**Observed behavior**:
- Loss numerically lower than pixel reconstruction (noise distribution is
  simpler), but this is not directly comparable.
- Loss dynamics are non-monotonic: drops sharply in early epochs, then
  **rebounds and plateaus around 0.98** for 200+ epochs.
- Interpretation: when curriculum reduces t (weaker noise), epsilon prediction
  gets harder (small noise is hard to precisely estimate amid large signal).
  So curriculum that was "easy→hard" for pixel prediction becomes "easy→hard
  again" for noise prediction, but in a different regime, causing the bounce.

**Limitation discovered**:
- Curriculum is not meaningful for epsilon prediction. Full-range uniform t
  sampling (standard DDPM) would be cleaner.
- Still does not address the "decoder ignores encoder" issue on diffusion
  downstream.

## 3. MAE-style masking (mae_masking)

**Config**: `pretrain_vit_b16_mae.yaml`
(`mae_masking: true`, `predict_noise: true`).

**Design (MaskDiT-inspired)**:
- Per-image t ~ Uniform[0, T) (standard DDPM, no curriculum).
- Noise added to all patches, then `mask_ratio` (default 0.25) of patches are
  randomly dropped entirely.
- Encoder only processes the visible (1 - mask_ratio) noisy patches
  (MAE-efficient: 25% fewer encoder FLOPs).
- Decoder inserts learnable mask tokens at dropped positions, predicts noise
  for all patches.
- Loss: `L_eps(visible) + 0.1 * L_eps(masked)`.

**Rationale**: combines MAE's global-reasoning inductive bias (mask forces
inference from partial context) with DDPM's epsilon target.

**Problem noticed (design-level, before full training)**:
- The 75% "visible" patches are all noisy. At high t, a visible patch and a
  masked patch carry equally little information.
- There is no "spatial anchor" at this configuration: the encoder has to infer
  from a context that is itself heavily corrupted.
- This loses the helpful structure signal that the original clean-patch anchors
  provided.

## 4. Dual decoder (current)

**Config**: `pretrain_vit_b16_dual.yaml` (`dual_decoder: true`).

**Design**:
- Encoder sees `clean_ratio` clean + `(1 - clean_ratio)` noisy patches
  (no MAE masking — clean patches return as spatial anchors).
- Per-image t ~ Uniform[0, T) (standard DDPM, no curriculum).
- Two decoders share the encoder:
  - `decoder` predicts noise epsilon (DDPM pathway).
  - `decoder_pix` reconstructs clean pixels (MAE pathway).
- Loss computed on the noisy patches only: `L_eps + pixel_loss_weight * L_pix`.

**Motivation**: explicit instantiation of the "framework" insight. Each
decoder represents one end of the MAE ↔ Diffusion spectrum. The shared encoder
is forced to learn a representation that serves both:
- Semantic content (for the pixel head).
- Noise-aware structure (for the epsilon head).

The clean anchors ensure the encoder always has at least some structural signal
to work from, even at high t.

**Expected downstream transfers**:
- Classification: pixel pathway's semantic bias helps; encoder features are
  high-level.
- Diffusion generation: epsilon pathway's decoder weights directly transfer;
  encoder provides conditioning that matches the downstream's epsilon target.

## Summary table

| Design | Clean anchors | t range | Target | Decoders | Status |
|---|---|---|---|---|---|
| 1. Original | yes (curriculum) | curriculum | pixel | 1 | done, +11% cls |
| 2. eps pred | yes (curriculum) | curriculum | epsilon | 1 | trained, rebound issue |
| 3. MAE-mask | no | full range | epsilon | 1 | not finalized (no anchor) |
| 4. Dual (early) | yes (fixed 25%) | full range | pixel + epsilon | 2 (4-layer each) | unstable, see stability.md |
| 5. Stability recipe | varies | varies | varies | 1 zero-init Linear head | **stable for 300+ epochs** |

## 5. Stability recipe (current working architecture)

After diagnosing and fixing the ε-prediction stability issues (see
[stability.md](stability.md)), the working architecture for all diffusion
variants is:

- **Encoder**: DiTEncoder with per-block adaLN-Zero time conditioning.
  All SubDiff diffusion variants (naive_ddpm, predict_noise, dual_decoder)
  use this, not the plain ViTEncoder.
- **QK-Norm**: LayerNorm per head on Q and K before the attention dot
  product. Prevents attention entropy collapse.
- **Minimal head**: single zero-init Linear (`embed_dim → patch_dim`)
  replacing the 4-layer Decoder transformer. The 4-layer Decoder, being
  time-unconditioned, washed out the carefully-learned time-dependent
  representations from the encoder.
- **Clean/noisy indicator embeddings**: learnable `(1, 1, D)` vectors
  added to patch tokens in SubDiff variants to tell encoder which patches
  are clean anchors vs noisy. Same additive injection as time embedding
  in ViTEncoder fallback, or as extra signal alongside adaLN in DiTEncoder.
- **Constant lr 1e-4**, no weight decay, no warmup, bf16 mixed precision,
  grad clip 1.0.

Under this recipe, all three diffusion variants (naive_ddpm_minimal,
eps_qknorm + DiT, dual + DiT) converge at essentially the same rate on
ε loss (~0.117 at end of epoch 0), with dual showing a tentative ~17%
advantage that may persist or disappear — see todo.md Priority 1.
