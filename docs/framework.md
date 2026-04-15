# Framework: MAE and Diffusion as One

## Core observation

MAE and Diffusion are not two separate paradigms. They are two extremes of the
same denoising autoencoder framework:

| Axis | MAE | Diffusion (DDPM) |
|---|---|---|
| Input corruption | Binary mask (info completely erased) | Gaussian noise (continuous strength) |
| Steps | Single-step recovery | Multi-step recovery (1000 typical) |
| Prediction target | Original pixel | Noise epsilon |
| Supervision | Clean image itself | Known noise from forward process |

Reading across the table: MAE is the extreme single-step case of diffusion —
masking is equivalent to "infinite-strength noise," and MAE does the entire
denoising in one shot. Diffusion is the continuous multi-step generalization,
with noise strength smoothly interpolating from near-zero to near-total.

## Implication 1: mask and noise live on the same spectrum

If mask is a limit of noise, then a patch at timestep `t` close to T behaves
almost identically to a masked patch: both contain no usable signal.

Consequence: the `clean_ratio` / `noisy_mask` split in the original SubDiff is
actually "discrete mask" layered on top of "continuous noise," and they can be
fluidly re-balanced. Pure MAE is one extreme (mask ratio 0.75, noise = infinity).
Pure DDPM is the other (mask ratio 0, noise = continuous t).

## Implication 2: curriculum = sliding along the spectrum

The original SubDiff curriculum decays timestep range from `[800, 1000]` to
`[100, 600]`. In this framework, that's exactly "start pretraining in the MAE
regime (strong noise ≈ mask), finish in the fine-denoising regime (weak noise ≈
ordinary diffusion)."

This reframe clarifies both when curriculum helps (transitioning representation
learning from coarse to fine) and when it hurts (if fine-denoising at low t
reduces to near-identity mapping and destroys the semantic features learned
early).

## Implication 3: downstream transfer has two channels

A pretrained encoder can feed two kinds of downstream tasks:

- **Discriminative** (classification, detection): wants high-level content
  understanding. The MAE pathway (single-step, pixel target) naturally produces
  this kind of semantic representation.
- **Generative** (diffusion models): wants features that condition a multi-step
  epsilon predictor. The DDPM pathway (multi-step, noise target) naturally
  aligns with this downstream objective.

If the pretraining objective is pure MAE, the discriminative transfer is strong
but generative transfer is weak (features are semantic but mismatched to the
epsilon-prediction decoder). If the pretraining objective is pure DDPM, the
reverse holds.

A pretraining framework that wants to serve both downstreams should train
**both objectives simultaneously**, sharing an encoder.

## Implication 4: the dual-decoder design

Taking Implication 3 seriously leads directly to the dual-decoder SubDiff:

- Encoder sees `clean_ratio` clean patches + `(1 - clean_ratio)` noisy patches
  (standard DDPM forward process, per-image t ~ Uniform[0, T)).
- `decoder` predicts the noise epsilon (DDPM pathway; output transfers to
  downstream diffusion).
- `decoder_pix` reconstructs the clean pixel values (MAE pathway; supervises
  encoder to learn semantic content).
- Loss: `L_eps + lambda * L_pix`, computed on the noisy patches only.

The two heads pull the encoder in different directions:
- Epsilon prediction pulls toward local noise-structure representations.
- Pixel prediction pulls toward global content understanding (especially at
  high t where the patch is pure noise and the only way to predict pixels is
  from spatial context).

The shared encoder is forced to represent both what kind of corruption is
present (epsilon pathway) and what content is underneath it (pixel pathway).

## Further generalization: parameterization equivalences

Within the diffusion family, there are multiple equivalent parameterizations of
the decoder's target:

- **epsilon prediction**: predict the noise directly.
- **x_0 prediction**: predict the clean image directly.
- **v prediction** (Salimans & Ho): a rotation of the two.

They are mathematically equivalent via the forward diffusion equation
`x_t = sqrt(alpha_cumprod) * x_0 + sqrt(1 - alpha_cumprod) * epsilon`, but
training dynamics differ:
- epsilon: variance is constant (N(0, I)), so loss is stable across t, but
  predicting near-zero at low t is wasteful.
- x_0: variance follows data distribution, which is harder at high t but more
  natural at low t.
- v: interpolates, generally smoother.

In our dual-decoder framing:
- `decoder` doing epsilon prediction is the diffusion pathway.
- `decoder_pix` doing x_0 prediction (conditioned on t) would also be a
  diffusion pathway, just a different parameterization.
- `decoder_pix` doing pixel reconstruction without t is the pure MAE pathway
  (equivalent to x_0 prediction at t = infinity).

This suggests a spectrum of dual-decoder variants: pure MAE + epsilon is one
end, dual epsilon-prediction + x_0-prediction (both with t) is the other, and
intermediate choices are possible.
