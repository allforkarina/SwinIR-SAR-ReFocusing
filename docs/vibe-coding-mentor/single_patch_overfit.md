# Scene 4 single-patch overfit diagnostic

## Purpose

This diagnostic asks one narrow question before any further long SAR training:

> Can the current complex SwinIR and optimization path memorize one explicitly
> selected Echo/Image pair?

It is not a generalization experiment. It deliberately repeats one sample and
uses the raw training model as the success authority. EMA metrics are recorded
only as supporting evidence because EMA can lag during a short run.

## Fixed sample and data contract

The agreed Scene 4 sample is:

```text
patch_row_17500_col_9400_2.mat
```

The Echo and Image files must have the same filename and each must contain one
finite complex `patch` matrix with the shape declared by the base configuration.
Both tensors use the existing paired normalization: one RMS scale is computed
from Echo and applied to Echo and Image.

The script verifies both files with SHA-256 and stores the hashes in the resolved
configuration and checkpoints. Resume is rejected if the sample content or any
resolved experiment setting changes.

## Run on the server

From the repository root:

```bash
python scripts/overfit_single_patch.py \
  --config configs/train_sar.yaml \
  --echo-file /home/sy/capella_scene4_paired/echo/patch_row_17500_col_9400_2.mat \
  --image-file /home/sy/capella_scene4_paired/image/patch_row_17500_col_9400_2.mat \
  --output-dir runs/scene4_single_patch_row17500_col9400 \
  --device auto
```

Defaults implement the agreed contract:

- full 512 by 512 real/imaginary input and output;
- current SwinIR architecture with `drop_path_rate=0`;
- complex Charbonnier loss;
- Adam with learning rate `2e-4`, betas `(0.9, 0.99)`, and no weight decay;
- constant learning rate for at most 5,000 optimizer steps;
- EMA decay `0.999`, reported but not used as the success authority;
- evaluation every 100 steps and artifact/checkpoint export every 500 steps;
- best-checkpoint comparison at exported checkpoints and the final/success step;
- early success only after three consecutive passing evaluations.

## Resume after interruption

Repeat the same command and add the checkpoint path:

```bash
python scripts/overfit_single_patch.py \
  --config configs/train_sar.yaml \
  --echo-file /home/sy/capella_scene4_paired/echo/patch_row_17500_col_9400_2.mat \
  --image-file /home/sy/capella_scene4_paired/image/patch_row_17500_col_9400_2.mat \
  --output-dir runs/scene4_single_patch_row17500_col9400 \
  --device auto \
  --resume runs/scene4_single_patch_row17500_col9400/checkpoints/latest.pt
```

All settings, including maximum steps and precision policy, must match the
original run. `interrupted.pt` is also written when Ctrl+C is handled.

## Success criteria

The raw model must simultaneously satisfy:

```text
normalized complex RMSE <= 0.10
complex coherence >= 0.95
magnitude correlation >= 0.95
prediction/target RMS ratio in [0.90, 1.10]
log-magnitude PSNR >= 30 dB
log-magnitude SSIM >= 0.95
```

PSNR and SSIM are computed on magnitude expressed relative to the one shared
Image target peak, clipped to `[-60, 0] dB`, then mapped to `[0, 1]`. SSIM uses
an 11 by 11 Gaussian window with sigma 1.5. They are explicitly not metrics on
the real and imaginary channels.

The report also contains zero-output and Echo-identity baselines. A lower
Charbonnier loss alone is not accepted as evidence of successful overfitting.

## Artifacts

Each output directory contains:

```text
resolved_config.json
metrics.jsonl
report.json
checkpoints/best.pt
checkpoints/latest.pt
checkpoints/final.pt
figures/step_*.png
predictions/step_*.mat
```

The figures compare Echo, raw prediction, EMA prediction, and Image target in
shared-reference log magnitude and phase. Prediction MAT files are converted
back to the original complex scale and contain `raw_prediction`,
`ema_prediction`, `normalization_scale`, and `step`.

## Interpretation

- **Pass:** the present model and training path can memorize this one pair. Move
  to a small set of spatially separated patches; do not claim generalization.
- **Raw passes but EMA lags:** the mapping is learnable in this controlled run;
  EMA decay or run length explains the lag.
- **Magnitude metrics improve but complex coherence does not:** investigate
  phase reference, phase ramps, or coordinate-grid differences.
- **Output RMS collapses while loss falls:** the earlier low-energy failure mode
  is reproduced even on one sample.
- **No pass by step 5,000:** do not launch a longer full-dataset run. Inspect
  spatial/phase alignment and whether a local patch contains the context needed
  for the PFA mapping.

## Verification scope

Local tests cover metric boundaries, success gating, a tiny CPU training run,
artifact production, and strict checkpoint restoration. The real 512 by 512 GPU
run and its scientific interpretation must be completed on the server data.
