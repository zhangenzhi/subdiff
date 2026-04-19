# Open Questions and Next Steps

## DONE — Stability fixes (iterations 1-5)

See `stability.md` for full journey. Summary:
- **QK-Norm** fixed attention entropy collapse (300+ epoch stability).
- **Minimal final head** (zero-init Linear, replacing 4-layer Decoder) fixed
  the "stable loss but pure-noise samples" problem — loss dropped from 0.34
  plateau to 0.05-0.12 and sampling produces real image structure.
- **DiTEncoder everywhere** (not just naive_ddpm) aligned SubDiff variants'
  convergence speed with naive DDPM.

Remaining stability tasks (low priority):
- EMA weights (decay 0.9999) for inference quality — standard practice,
  expected to improve FID but not critical for current training.
- Logit-normal t sampling (SD3-style) instead of uniform — may help mid-t
  learning efficiency.

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
