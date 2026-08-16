# Coordinate-conditioned 16-patch diagnostic

## Evidence motivating the experiment

The four persistent joint-run failures can each pass when trained alone. Full-set
gradient accumulation stabilizes the raw model at 12 of 16 but plateaus after 2,000
updates. The remaining mapping conflict is therefore consistent with missing global
context rather than invalid pairs or simple optimizer ordering.

## Experiment

- Load the raw weights from the best full-set consolidation checkpoint.
- Reuse and hash-verify its exact 16-patch manifest.
- Normalize each patch start row and column to `[-1, 1]` using the selected scene
  bounds. The selected set includes the scene-grid extrema.
- Map the two coordinates through a 2-layer MLP and produce one scale and bias per
  shallow feature channel.
- Apply FiLM immediately after `conv_first`.
- Zero-initialize the final coordinate layer so epoch 0 is exactly equivalent to the
  unconditioned source model.
- Use learning rates `1e-5` for the existing SwinIR and `5e-4` for the new coordinate
  branch, with one full-set mean-gradient update per epoch.
- Evaluate every 50 epochs and require raw 16/16 success three times consecutively.

## Interpretation

- **16/16 pass:** the local Echo patch is insufficient without its parent-image
  location. Full training should provide global coordinates and, across parent
  images, scene identity or acquisition geometry.
- **Stable 12/16:** shallow coordinate FiLM is insufficient; test larger spatial
  context before increasing generic network capacity.
- **Regression:** coordinate modulation or its learning-rate separation is harmful;
  retain the unconditioned consolidation checkpoint.

## Verification and ownership

Tests prove exact zero-initialized equivalence at multiple coordinates, coordinate
normalization, coordinate-branch learning, one optimizer update per complete set,
checkpoint output, and resume behavior. The real CUDA run is still required. This is
an AI checkpoint; developer mastery validation remains pending by prior delegation.
