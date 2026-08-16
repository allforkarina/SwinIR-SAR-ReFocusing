# Full-set gradient consolidation diagnostic

## Problem

All four patches that failed the 16-patch joint run can satisfy the strict criteria
when trained individually. The joint raw model also oscillated between 0 and 6
passing samples near the end, while its EMA model reached 12 of 16. These observations
support destructive interference between per-patch optimizer updates.

## Experiment

- Load the `ema_model` weights from the completed 16-patch run.
- Reuse the exact embedded sample manifest and verify every MAT-file hash.
- Initialize a new Adam optimizer; do not reuse moments associated with the raw model.
- Process all 16 patches with physical batch size 1, divide each loss by 16, and
  accumulate their gradients.
- Perform exactly one optimizer update after the complete set.
- Use learning rate `5e-5`, EMA decay `0.99`, and at most 2,000 full-set epochs.
- Evaluate every 50 epochs and archive every 200 epochs.
- Require the raw model to pass all criteria on all 16 patches for three consecutive
  evaluations.

This has the optimization behavior of a batch containing all selected patches without
requiring all 16 full-resolution activations to coexist in GPU memory.

## Interpretation

- **16/16 pass:** sequential single-patch updates were the dominant failure mode.
  Effective-batch accumulation and EMA should be evaluated before full-dataset
  training.
- **Stable improvement without 16/16:** increase effective diversity gradually and
  inspect which metrics remain limiting before changing model capacity.
- **Regression from the source EMA:** the averaged gradient or learning rate is not
  preserving the useful compromise; stop rather than scaling the experiment.

## Verification and ownership

Tests verify deterministic complete-set ordering, evaluation-budget validation, EMA
initialization metadata, one Adam update per full-set epoch, artifacts, checkpoints,
and resume behavior. A real CUDA run remains required. Implementation is an AI
checkpoint because the developer delegated coding and will pull and run it remotely;
developer mastery validation remains pending.
