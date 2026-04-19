# Training Stability: ε-Prediction on ViT in Pixel Space

Running document of the stability journey: what broke, what the hypotheses were,
what actually fixed each failure mode, and the (tentative) positive finding at
the end.

## Timeline summary

| Iteration | Architecture change | Outcome |
|---|---|---|
| 1 | Plain ViT encoder + 4-layer Decoder, eps prediction | Loss reaches 0.34 plateau, then diverges around epoch 10-25 |
| 2 | + AdaLN-Zero in encoder (DiTEncoder, per-block time conditioning) | Delays divergence to epoch 43 but still collapses |
| 3 | + QK-Norm (LayerNorm on Q/K per head, ViT-22B/SD3 style) | **Stable for 300+ epochs**, loss plateau at 0.34 |
| 4 | + Replace 4-layer Decoder with single zero-init Linear head | **Loss drops to 0.05-0.12**, sampling produces image structure (was pure noise before) |
| 5 | Apply DiTEncoder (not just ViTEncoder) to all diffusion variants | SubDiff eps/dual now converge at same rate as naive DDPM |
| 6 | Observed in dual-decoder: pix head helps ε head | dual ε loss 0.10 vs naive/eps-only 0.12 (~17% gap) |

## Failure mode 1: ε-prediction loss divergence (iterations 1-2)

### Observation

Across four independent runs with different configs, the ε-prediction loss
followed the same pattern:

1. Healthy descent for 4-15 epochs (loss 0.9 → 0.3)
2. Long plateau at 0.30-0.40
3. Gradual ramp-up over ~5 epochs
4. Lock-in at the predict-zero baseline (loss ≈ 1.0), very low variance

| Run | Config | Divergence starts |
|---|---|---|
| eps-only curriculum | predict_noise + curriculum t | epoch 10 |
| dual_uniform | dual-decoder + uniform t | epoch 15 |
| dual_curriculum | dual-decoder + curriculum t | epoch 24 |
| naive_ddpm (cls-token time inject) | additive time to cls | epoch 9 |
| naive_ddpm (adaLN-Zero per block) | DiT-style encoder | epoch 43 |

### Correct diagnosis: attention entropy collapse

The ramp-then-plateau at predict-zero baseline is the canonical signature of
**attention entropy collapse**: Q and K norms grow unboundedly, softmax becomes
one-hot, gradients become pathologically imbalanced, effective model capacity
collapses, and the network outputs the trivial zero prediction.

### Fix: QK-Norm

LayerNorm on Q and K per head before the dot product. Bounds attention logits
and prevents entropy collapse. This is standard in ViT-22B, SD3, PixArt-Σ,
FLUX — we should have had it from the start.

With QK-Norm, naive DDPM-ViT trained stably for 300+ epochs at loss=0.34 with
no divergence.

## Failure mode 2: stable loss but pure-noise samples (iterations 3-4)

### Observation

Post QK-Norm, the naive DDPM-ViT trained happily with loss=0.34 for hundreds of
epochs. But multi-step DDPM/DDIM sampling produced **pure colored noise**, no
image structure at all. The x_t range during sampling was also abnormal (±5
instead of expected ±3).

Worth noting: single-step reconstructions during training visualization were
reasonable — the model could denoise a known-t noisy image back to something
with structure. But from-scratch generation failed completely.

### First attempts that didn't fully fix it

- Fixed DDIM strided sampling formula (old code assumed 1-step DDPM even with
  stride=4). Got numerical range saner (±5 → ±4) but still pure noise.
- Switched to constant lr: didn't help.

### Correct diagnosis: time-conditioning dilution in the 4-layer Decoder

Our architecture had:

```
DiTEncoder (12 layers, per-block adaLN with time c)
     ↓  patch tokens carry time-conditioned information
Decoder (4 plain ViT blocks, NO time conditioning!)
     ↓  12 layers of time-independent self-attention
     ↓  "blurs" time-conditioned representations across t
Linear head
```

The 4-layer plain Decoder acted as an **"information black hole"** for the time
signal. Without time conditioning, each decoder block does "cross-t averaged"
spatial mixing, washing out the carefully encoded t information from the
encoder. The model learns "predict-mean" per patch — which minimizes MSE
adequately for training (loss ≈ 0.34) but produces garbage when iterated over
250 sampling steps because ε̂ at each t is a mixture of cross-t averages rather
than the specific ε for the current t.

**This also explains why training loss was stable but high at 0.34**: that's
the minimum achievable by a model that cannot disambiguate t across its
internal representation — essentially the conditional mean E[ε | x_t] ignoring
t.

### Fix: DiT-style minimal final head

Replace the 4-layer Decoder with a single zero-initialized Linear head, exactly
like DiT's `FinalLayer.linear`. The encoder's adaLN_final already provides
time-conditioned LayerNorm; the Linear maps directly to patch_dim eps.

```python
self.decoder = nn.Linear(embed_dim, patch_size**2 * 3, bias=True)
nn.init.zeros_(self.decoder.weight)
nn.init.zeros_(self.decoder.bias)
```

**Result**: loss drops from 0.34 plateau to 0.05-0.12 within 1 epoch. Step 0
loss = 1.0000 exactly, matching the zero-init predict-zero baseline. Epoch 5
samples show clear per-patch structure (not pure noise), though with visible
patch grid (to be refined by continued training).

## Failure mode 3: SubDiff variants trained slower than naive DDPM (iteration 5)

### Observation

After the minimal-head fix, three parallel runs had noticeably different
ε-prediction convergence speed:

| Run | Epoch 1 ε loss | Encoder |
|---|---|---|
| naive_ddpm_minimal | 0.06-0.08 | DiTEncoder (per-block adaLN) |
| eps_qknorm + minimal head | 0.14 | ViTEncoder (no adaLN) |
| dual + minimal head | 0.15 | ViTEncoder (no adaLN) |

### Diagnosis: encoder-level time dilution in SubDiff variants

SubDiff variants used `ViTEncoder` (plain ViT blocks) while naive_ddpm used
`DiTEncoder` (adaLN per block). In the ViTEncoder path, time was injected
additively **only once** before the first transformer block, then 12 layers of
time-independent self-attention further diluted it — the same time-dilution
problem, now at the encoder.

### Fix: use DiTEncoder everywhere

Changed encoder selection from `if naive_ddpm` to
`if naive_ddpm OR predict_noise OR dual_decoder`. All diffusion variants now
use DiTEncoder with per-block adaLN-Zero.

Added `DiTEncoder.forward_patches(patch_tokens, c)` that separates patch_embed
from the rest of the forward. SubDiff's `_encode_with_indicators` can now:

1. patch_embed(image)
2. add clean/noisy indicator per patch
3. call `encoder.forward_patches(patched_plus_indicator, time_emb)` — time
   enters via per-block adaLN-Zero.

**Result**: SubDiff eps_qknorm and dual now converge to loss 0.12 at end of
epoch 0, matching naive_ddpm exactly. The 2× slowdown from before was
entirely explained by the encoder architecture mismatch.

## Retracted: "pix head helps ε head" (iteration 6 — self-correction)

### What we initially saw

With architecture fully aligned across three runs (DiTEncoder + adaLN-Zero +
QK-Norm + minimal head + same lr schedule), epoch-0 ε loss was:

| Run | Epoch 0 end ε loss |
|---|---|
| naive_ddpm_minimal | 0.117 |
| eps_qknorm + clean anchor | 0.117 |
| dual + clean anchor + pix head | 0.099 |

This 17% advantage for dual was initially interpreted as "pix head positively
transfers to ε head via the shared encoder." **That interpretation was wrong**
— see iteration 7 below. The lower ε loss in dual turned out to be a
superficial indicator masking a mode-collapse failure at sampling.

## Failure mode 4: pix head causes mode collapse at sampling (iteration 7)

### Observation

Multi-step DDIM sampling of the three models at matched architecture:

| Run | Per-patch pixel std | ||image - tiled_mean|| / ||image|| |
|---|---|---|
| naive_ddpm (ep 74) | 30.7% | 56.8% |
| eps_clean (ep 69) | 41.5% | 65.6% |
| **dual_clean (ep 10)** | **4.3%** | **9.2%** |

Interpretation of the second column: we split the generated image into 196
patches, compute the mean patch, tile it 14×14, and measure how close the
actual image is to this tile. **9% error means the image is almost entirely
one mean patch repeated**. naive and eps have ~60% error (patches are largely
independent).

**Dual's generated image is essentially the same 16×16 template tiled 196
times**, with tiny (4.3%) per-position perturbation.

### Diagnosis: OOD mode collapse driven by pix head

Training distribution:
- 25% of patches are clean ImageNet pixels (real content, structured)
- 75% are noisy at random t
- Encoder learns to use BOTH content (from clean anchors) AND position

Sampling distribution (OOD):
- 100% of patches are pure Gaussian noise
- No clean anchors, no real content signal anywhere
- Encoder's content pathway sees "nothing informative"

When content signal is uninformative, encoder falls back to its strongest
remaining signal: the **pix head's training-time prior**, which trained the
encoder to output features close to "average natural image patch statistics."
This prior is approximately position-invariant (any given position's mean
over all ImageNet images is similar to any other position's mean), so:

1. Every position's encoder output converges toward the same representation.
2. Decoder maps this to the same "mean patch" at every position.
3. DDIM iteration amplifies this (attractor dynamics over 250 steps).
4. Final image = tiled mean patch + 4% noise-driven perturbation.

This is a classic **mode collapse**: maximum sample diversity is lost. All
16 "different" samples from dual produce nearly identical tiled outputs.

### Why naive doesn't collapse

naive's encoder was trained only on "all noisy" inputs — sampling is
in-distribution. Encoder relies on the random per-patch content signals
(different for each position at sampling). Result: different predictions
per position, avoiding mode collapse. The price is no cross-patch
coherence — patches are independent and visually incompatible.

So naive fails by **structural incoherence** (every patch does its own
thing). dual fails by **information loss** (every patch does the same
thing). Information loss is strictly worse for a generative model:
naive's output is at least an injective map from noise to output, while
dual's output ignores the initial noise almost entirely.

### Why eps_clean doesn't collapse but still fails

eps_clean has clean anchors (like dual) but NO pix head. The encoder never
learns an "average natural image" target — only ε prediction. At sampling:
- All-noisy input is still OOD (same as dual)
- But encoder has no image-prior to collapse to
- Instead it produces "high-variance noise predictions" per patch (41.5%
  per-position std)
- Result: wild snow-like noise output, no coherence

So only **dual's pix head is the mode-collapse trigger**. eps_clean trades
mode collapse for raw ε prediction divergence.

### Revised understanding of SubDiff's dual design

Previous (wrong) reading:
- Dual's 17% lower ε loss = "pix head helps ε head"
- Implied dual pretraining accelerates both discriminative AND generative
  downstream

Corrected reading:
- Dual's lower ε loss = pix head pulls the encoder toward the natural-image
  manifold, which reduces ε prediction variance ON AVERAGE but at the cost
  of generative diversity
- For **discriminative transfer** (classification +11% top1) this is great:
  the encoder learns rich content representations
- For **direct generation** this is catastrophic: the learned content prior
  becomes an OOD attractor at sampling time

This matches the pattern of DINO / MAE: excellent SSL pretrainers, but not
directly usable as generative models. dual fits the same archetype.

### What this means for the SubDiff story

- **Discriminative downstream** (classification, detection, segmentation):
  dual is the right choice. Pix head's image prior = rich semantic features.
- **Generative downstream**: dual cannot be used directly. Must either:
  1. Finetune without pix head (and without clean anchor) to "un-collapse"
     the encoder — essentially a domain adaptation step.
  2. Apply CFG-style training: 10-20% of training batches use clean_ratio=0
     so encoder sees "all-noisy" inputs and doesn't build an OOD gap.
  3. Reframe SubDiff as a representation-learning method only; use a
     separate model for generation.
- The "single pretraining for both judgment and generation" narrative is
  **not supported** by the data. Dual is a representation learner.

## Current verified stability recipe

For stable ε-prediction on ViT in pixel space (unconditional, ImageNet):

- Architecture:
  - DiTEncoder with per-block adaLN-Zero (zero-init modulation)
  - QK-Norm (LayerNorm per head on Q and K before dot product)
  - Single zero-init Linear head (no deep decoder)
- Optimizer:
  - AdamW, lr=1e-4 constant, no weight decay, no warmup
  - bf16 mixed precision, gradient clip 1.0
- Diffusion:
  - Linear beta schedule 1e-4 → 2e-2, 1000 steps
  - Uniform t sampling
- Sampling:
  - DDIM (η=0) or DDPM ancestral (η=1), stride allowed
  - Use `alpha_bar[ts[i+1]]` for strided DDIM (not `alpha_bar[t-1]`)

## Open questions

1. Is the dual finding real and reproducible? (priority 1, see todo.md)
2. What is the downstream FID advantage (if any) of dual over naive when used
   as pretraining for a longer diffusion finetune?
3. Does the pix head's positive contribution to ε learning persist throughout
   training, or is it only early?
4. If we use continuous time instead of discrete (as in flow matching / EDM),
   does the dual advantage grow or shrink?
