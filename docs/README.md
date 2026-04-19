# SubDiff Research Notes

Self-supervised visual pretraining that unifies masked reconstruction (MAE-style)
and diffusion noise prediction (DDPM-style) in a single framework.

## Table of Contents

1. [framework.md](framework.md) — Theoretical framing: MAE and Diffusion as two
   ends of the same denoising autoencoder spectrum.
2. [designs.md](designs.md) — Evolution of design choices: original SubDiff →
   noise prediction → MAE-masked → dual-decoder.
3. [experiments.md](experiments.md) — Experiment log with quantitative results.
4. [stability.md](stability.md) — Training stability analysis: why ε-prediction
   on plain ViT diverges, attention entropy collapse hypothesis, and planned
   QK-Norm + EMA fixes.
5. [todo.md](todo.md) — Open questions and future directions.

## One-sentence summary

Every design in this project is an instance of "input is corrupted (by mask or by
noise), encoder processes what remains, decoder predicts the corruption (pixels
or epsilon)" — the decision is which corruption, how much, and what target.
