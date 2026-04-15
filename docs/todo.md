# Open Questions and Next Steps

## Priority 1 — Does the diffusion decoder actually use encoder conditioning?

Central question for the generative-downstream story. Unclear from aggregate
loss alone.

### Proposed diagnostic: per-t loss bucketing
Instrument `finetune_diffusion.py` to log loss separately for t in
`[0, 200), [200, 500), [500, 800), [800, 1000)`. Predictions:
- If pretrained encoder helps, the **high-t buckets drop faster** than scratch
  (these are regimes where the noisy patch alone is uninformative).
- If the decoder has learned a shortcut, all buckets look similar.

Cost: a few lines in the forward pass, and tensorboard writes.

### Proposed ablation: strip encoder conditioning
Add a flag that zeros out encoder features (or passes random vectors). If the
loss curve stays the same, the decoder was already ignoring encoder.

### Architectural intervention if needed
- Encoder feature dropout during training (force decoder to tolerate missing
  conditioning, which paradoxically makes it *rely* on the conditioning
  signal when available).
- Weight loss by `t` so high-t bins matter more (where conditioning is
  irreplaceable).
- Classifier-free guidance: randomly drop encoder conditioning with 10%
  probability, which doubles as stronger utilization forcing.

## Priority 2 — Confirm dual-decoder actually learns both objectives

Need to verify that the shared encoder is not collapsing to just one pathway.

### Diagnostics to add
- Log `L_eps` and `L_pix` separately (already done via `noisy_loss` and
  `clean_loss` fields).
- Compute gradient norm per decoder and per encoder block. If one decoder's
  gradients dominate, the other is not really training.
- Reconstruction visualizations side-by-side (already added in `_visualize_dual`).

### If one pathway dominates
Adjust `pixel_loss_weight`. Current default is 1.0; might need to tune to 0.5
or 2.0.

## Priority 3 — x_0-prediction variant of pixel decoder

From framework.md section "parameterization equivalences":

Make `decoder_pix` condition on `t` and predict x_0 (clean pixels adjusted for
the timestep). This brings both heads firmly into the diffusion family, with
epsilon-pred and x0-pred being complementary parameterizations.

Implementation:
- Add a simple time embedding MLP to `Decoder` (optional input).
- In `_forward_dual`, pass `t` to `decoder_pix`.
- Downstream: can sample from either parameterization, can even ensemble them.

## Priority 4 — Curriculum-less training vs curriculum training

Run 2 showed that epsilon prediction with curriculum has weird loss dynamics.
The dual-decoder run uses no curriculum (full-range t). Worth a direct A/B:
same dual-decoder design, one with curriculum-restricted t, one with full
range, compare downstream performance.

## Priority 5 — MAE-masking + clean anchors + dual decoder

The natural combination of everything: some patches masked (no info), some
clean (anchors), rest noisy (main signal). Three-way split:
- ~25% masked
- ~25% clean anchor
- ~50% noisy
Both decoders (eps, pix) predict for the masked + noisy portions. Tests
whether harsher corruption plus anchors plus dual targets pushes the encoder
to a stronger representation.

Implementation: extend the 2-way split in `_forward_dual` to a 3-way split.
Encoder uses `forward_masked` (MAE-efficient) with the 25% masked portion.

## Priority 6 — Longer/larger scale

All current runs are 300-epoch pretraining on ImageNet-1K, limited by the
16-hour walltime (requires one resume). For a publishable result we would
want:
- 800 epochs (MAE-standard) for fair comparison to prior work.
- linear probe numbers (not just finetune), which is the standard benchmark.
- ImageNet-22K pretraining (if data is available) for scale comparison.

## Priority 7 — Comparison baselines to include in any writeup

- MAE (pure pixel reconstruction, 75% mask) reimplemented on same codebase.
- MaskDiT (masked diffusion) on same codebase.
- DINO or iBOT using a public checkpoint, then our downstream diffusion
  finetune pipeline, to check if DINO features really do accelerate
  generation (the original motivation from RAE-style papers).

## Lower priority / parking lot

- FID / generation quality metrics once diffusion finetune converges.
- Cross-attention visualization in the downstream diffusion decoder.
- Different encoder scales (ViT-S, ViT-L) to check how the story scales.
- Continuous noise schedule (flow matching or rectified flow) instead of DDPM.
