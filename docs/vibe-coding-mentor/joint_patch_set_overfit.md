# Joint 16-patch overfit diagnostic

## Problem

One Scene 4 pair can be memorized by the current SwinIR and loss, but that does not
show whether a single model can represent several spatially separated Echo-to-Image
mappings. This experiment is the next diagnostic between one-pair overfitting and
full-dataset training.

## Design

- Start with `patch_row_17500_col_9400_2.mat`, the pair that passed the single-patch
  test.
- Select 15 additional pairs with deterministic normalized farthest-point sampling.
- Reject candidates whose 512x512 source rectangles overlap any selected rectangle.
- Normalize every pair independently using Echo RMS, exactly as in the training data
  pipeline.
- Train with physical batch size 1 and one deterministic shuffle per 16-sample epoch.
- Evaluate raw and EMA models on all 16 samples every 1,600 optimizer updates.
- Treat the raw model as authoritative. A run passes only when all 16 samples satisfy
  every single-patch threshold for three consecutive evaluations.
- Save the anchor and current worst-RMSE sample as representative artifacts. Record
  all per-sample metrics in JSONL and the final report.

The defaults allow 50,000 updates, or 3,125 sample presentations per patch. This is
close to the 2,800 updates required by the successful one-patch run while allowing
for transfer between samples.

## Interpretation

- **16/16 pass:** the model and objective can jointly represent a small spatially
  diverse set. The next experiment should increase the patch-set size before full
  training.
- **Some samples pass:** inspect the reported worst samples and spatial pattern. This
  supports position/content-dependent conflict rather than a broken optimizer.
- **0/16 or little improvement:** compare the selected-set baselines and verify the
  chosen Echo/Image stage relationship before scaling training.

EMA is reported but cannot make the run pass. With decay 0.999 it is expected to lag
the raw model during a finite overfit diagnostic.

## Verification and ownership

Automated tests cover deterministic non-overlapping selection, shuffled epoch
coverage, argument validation, artifacts, checkpoints, and resume behavior. The
experiment still requires a CUDA server run on the real Scene 4 MAT files.

The developer explicitly delegated implementation and push because they can only
pull and run on the server. This checkpoint therefore records code implementation;
developer mastery validation remains pending.
