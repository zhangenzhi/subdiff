# Open Questions and Next Steps

## DONE — Stability fixes (iterations 1-5)

See `stability.md` for full journey. Summary:
- **QK-Norm** fixed attention entropy collapse (300+ epoch stability).
- **Minimal final head** (zero-init Linear, replacing 4-layer Decoder) fixed
  the "stable loss but pure-noise samples" problem — loss dropped from 0.34
  plateau to 0.05-0.12 and sampling produces real image structure.
- **DiTEncoder everywhere** (not just naive_ddpm) aligned SubDiff variants'
  convergence speed with naive DDPM.
- **patch_size=2 at 32×32** (Run 8) confirmed that patch size is the
  core bottleneck for pixel-space ViT diffusion at 224×16.
- **Rectified Flow path** (Run 9) implemented and trained: v-pred + logit-
  normal t. Trains stably but does not fix the tiled-patch failure at 224×16.
- **Logit-normal t sampling** (SD3-style) — shipped as part of the RF path;
  also available via `rf_t_sampling` flag.

Remaining stability tasks (low priority):
- EMA weights (decay 0.9999) for inference quality — standard practice,
  expected to improve FID but not critical for current training.

## DONE — Run 10 RF + MAE mask validated

Run 10 trained 169 ep (24h walltime, 2 GPU). visible v-loss 0.187 at
ep 163 ≤ Run 9 ep 299 best (0.191). 50-step samples match Run 9 ep 299
quality. Single-pass MAE inpainting works (`scripts/inpaint_rf_mae.py`).
Verdict: **MAE auxiliary task accelerates RF convergence by ~45% epochs
to the same quality.** First clear architectural win at 224×16 pixel
space.

## DONE — Run 11 (p16) and Run X (p8) dual-head RF for inpainting

- Run 11: 300 ep on 4 GPU done 2026-04-28, final avg_loss 0.259.
- Run X: 300 ep on 16 GPU in progress (271/300), best 0.149 at ep 269.
- Inpaint at t=1 + prompt=25% works: model recovers 75% missing pixels
  from 25% context, no mode collapse (each row preserves identity).
- Head-to-head at matched wall time (~22h): p8 wins on both loss (-42%)
  and perceptual sharpness (scales/logos/faces visible vs blurred blobs).
- See `experiments.md` Run 11 / Run X sections and takeaway #8.
- Two-stage `inpaint_rf_dual.py` (x_0-head init + Euler refinement) not
  yet built — Run X single-pass is already strong, iterative refinement
  is now an *enhancement*, not a *necessity*.

## DONE — Run 12 cold-RF Refiner validates μ→x_0 chain

Trained 100 ep on 16 GPU (~10h). Refiner v-head only, frozen Run X
ep 299 supplies μ. Loss plateau ~0.0176.

K-step Euler inference recovers high-frequency texture monotonically:
- K=1  → 17% target HF energy
- K=32 → 64% target HF energy (recommended)
- K=64 → 70%, converged (no overshoot to noise)

pixel_MSE INCREASES with K (μ wins on L2 *by being blurry*). The right
metric is HF energy ratio. Visually K=32 has photographic textures
(scales, fabric, skin) where μ has smooth blobs. See experiments.md
Run 12 section + takeaway #9 for the full table.

## Priority 0 — Pick the next experimental direction (open)

### A. Perceptual metric quantification (LPIPS) — fastest credibility win
- Compute LPIPS(prediction, ground_truth) on 1000+ val images for
  Run X μ (single pass) vs Run 12 K=32. Expected: Run 12 wins
  decisively, mirroring the visual gap.
- Cost: ~10 GPU-min. Gives a defensible perceptual number for the
  writeup, complementing pixel_MSE / HF_energy.
- Why first: pixel_MSE looks bad for Run 12, we need a metric that
  both correlates with perception AND isn't trivially the chosen
  diagnostic. LPIPS is the field standard.

### B. FID for inpainting
- 5000 samples from ImageNet val, p_ratio=0.25, t=1.0 inpaint.
- Compare Run 10 / Run 11 / Run X (μ) / Run 12 (K=32) on matched seeds.
- Cost: 4-8 GPU-hours per model.

### C. Run 12 LR/training tweaks for higher HF recovery
- Refiner converged at 70% target HF energy. Possible ways to push
  past: longer training (200 ep), larger batch, lower LR cosine,
  uniform-t sampling (cover t=1 endpoint better at training), or
  larger Refiner.
- Each is cheap (Refiner trains in 10h on 16 GPU); pick whichever
  has a clean theoretical motivation.

### D. Downstream cls / diffusion finetune on Run X encoder
- Reconfirms Priority 1 / 1d finding with the strongest encoder we
  have. Closes the SubDiff "pretrain → downstream" arc.

### E. patch_size=4 on 32 GPU
- Defer; Run 12 shows the bottleneck has moved from "patch expressivity"
  to "irreducible HF given 25% context." Smaller patches help the former,
  not the latter.

## Priority 1 — Verify the "pix head helps ε head" finding

Epoch 0 data shows dual (DiT + clean anchor + ε + pix heads) has 17% lower
ε loss than naive DDPM with identical architecture. This is the only
finding where SubDiff's design helps the pretraining task itself (not just
downstream). Needs rigorous validation:

### 1a. Persistence across training
Keep running all three runs (naive, eps, dual) to epoch 30/50/100 and
compare ε loss at matching epochs. If dual's advantage disappears by epoch
50, the finding is noise, not signal.

### 1b. Reproducibility across seeds
Current data is from a single seed per run. Re-run naive and dual with 2-3
additional seeds. Expected seed-to-seed variance < 5%; if the 17% gap holds
across seeds, it's real.

### 1c. Compute-matched comparison
Dual does ~20% more compute per step (two heads + two loss terms). For a
fair comparison at equal compute budget, either:
- Train dual for proportionally fewer epochs.
- Or normalize by wall-clock rather than epoch count.

### 1d. Downstream diffusion finetune
If the ε advantage in pretraining is real, initializing a diffusion
finetune from dual should converge faster than initializing from naive.
Direct test of "dual pretraining accelerates diffusion training."

## Priority 2 — Generation quality evaluation

Once one of the stable runs has trained long enough (50-100 epochs):

1. Multi-step DDIM/DDPM sampling (5000 samples, ~2K+ needed for FID)
2. FID against 5000-image ImageNet val reference (already prepared in
   `fid_reference/`)
3. Compare: naive_ddpm_minimal vs eps_dit vs dual_dit
4. Visual comparison of 16-sample grids (current sampling at epoch 5 shows
   patch-level structure, should become more coherent at epoch 50+)

Target: honest baseline FID numbers. Expectations are modest — pixel-space
ViT-B without class conditioning or EMA is not going to hit SOTA; a number
in the 50-100 range is reasonable. The question is **relative** differences
between the three runs.

## Priority 3 — Downstream classification (reconfirm with new setup)

Old Run 1 (pixel-recon pretrain) gave +11% top1 on cls finetune.
With the new architecture (DiTEncoder + minimal head), re-run cls finetune
on:
- naive_ddpm_minimal (pure ε pretrain, no clean anchor) — should be weaker.
- eps_dit (clean anchor, single ε head) — should be mid.
- dual_dit (clean anchor, ε + pix heads) — should be strongest (replicates
  old Run 1 observation with cleaner architecture).

If dual > eps > naive on cls finetune top1, we have a clean multi-point
ordering showing the value of each component.

## Priority 4 — Diffusion-specific eval & ablation

If dual pretraining accelerates diffusion finetune (Priority 1d confirmed):
- Ablate the indicator embeddings (do they help? or is it only the pix
  head that matters?).
- Ablate clean_ratio (does 25% matter, or is any anchor fraction OK?).
- Test varying pixel_loss_weight.

## Lower priority / parking lot

- FID / generation quality metrics once diffusion finetune converges.
- Cross-attention visualization in the downstream diffusion decoder.
- Different encoder scales (ViT-S, ViT-L) to check how the story scales.
- Continuous noise schedule (flow matching or rectified flow) instead of DDPM.
- MAE-masking + clean anchors + dual decoder: three-way patch split
  (masked + clean + noisy) with MAE efficiency. Requires non-trivial code
  refactor; only worth doing if Priority 1 finding holds.

## Parking lot (older items, most subsumed by current priorities)

- Per-t loss bucketing to see whether pretraining helps specific t regions —
  useful if Priority 1d shows downstream diffusion speedup, to localize where
  the benefit comes from.
- x₀-prediction variant of pixel head (decoder_pix conditioned on t) — would
  unify the dual-decoder into two diffusion parameterizations.
- MAE-masking + clean anchors + dual decoder (3-way patch split). Worth
  revisiting only if Priority 1 finding is confirmed.
- 800-epoch pretraining on ImageNet-22K for publication-grade numbers.
- Compare against external baselines: MAE (reimplemented), MaskDiT, DINO
  checkpoints — needed for writeup.
