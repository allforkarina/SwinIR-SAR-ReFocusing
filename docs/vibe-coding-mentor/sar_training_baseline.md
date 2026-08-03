# SAR Training Baseline

## Scope

This milestone adds the paired MATLAB Dataset and a single-process,
single-GPU training entry point for the existing same-size SwinIR model. It
does not add inference or an independent `test.py`.

## Confirmed data contract

- Each echo and image MAT file supplies one complex `patch` matrix of 512 by
  512 pixels.
- Filenames encode a complete 95 by 172 coordinate grid with 100-pixel steps.
- The validation rectangle is row 3301--6201 and col 8200--13500.
- The guard rectangle is row 2801--6701 and col 7700--14000.
- Expected counts are train 13,780, guard 940, validation 1,620.
- Each pair uses only its echo RMS for scale:
  `sqrt(mean(abs(echo)^2) + 1e-12)`.

## Training contract

- Input and target are real/imaginary float32 tensors shaped `[2, 512, 512]`.
- Loss is joint complex Charbonnier with epsilon `1e-3`.
- Adam: lr `2e-4`, betas `(0.9, 0.99)`, no weight decay.
- MultiStepLR milestones: 75k, 120k, 135k, 142.5k; gamma `0.5`.
- EMA decay is `0.999`; EMA is validated and used for best checkpoints.
- One physical sample and one optimizer update per successful global step.
- Precision is CUDA BF16 when supported, CUDA FP16 with GradScaler otherwise,
  and CPU FP32 for smoke tests.

## Artifact and recovery contract

`runs/<run-name>/` contains the resolved YAML, JSON split manifest, JSONL
metrics, TensorBoard logs, a text log, and checkpoints. `latest.pt` is atomically
replaced on every validation interval; `best.pt` tracks the lowest validation
Charbonnier result; checkpoints are archived every 25k steps. Resume requires
an exact config and manifest fingerprint match.

## Developer-owned work

The developer implemented and corrected:

1. coordinate classification with validation priority over guard;
2. echo-RMS paired complex normalization;
3. one training step, including loss validation, mixed precision handling,
   scheduler advancement, and EMA update ordering.

## Verification

- Dataset/unit coverage: pairing, spatial split, normalization, sampler.
- Training primitive coverage: complex loss, RMSE, EMA.
- Training-step coverage: successful CPU optimizer update, scheduler update,
  and EMA update.
- Full local suite: 48 tests passing at the point this document was written.

## Known boundary

The local workspace does not have the optional `tensorboard` package installed,
so a real `run_training` invocation requires `pip install -r requirements.txt`.
No full 512-pixel server-data run has been performed locally.
