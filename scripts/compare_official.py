"""Compare this implementation with a fixed official SwinIR source file."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from swinir import DropPath, SwinIR, to_2tuple


SMALL_CONFIG = {
    "img_size": 8,
    "patch_size": 1,
    "in_chans": 2,
    "embed_dim": 12,
    "depths": [2, 2],
    "num_heads": [3, 3],
    "window_size": 4,
    "mlp_ratio": 2.0,
    "drop_path_rate": 0.0,
    "upscale": 1,
    "upsampler": "",
}


def _install_timm_compatibility_shim() -> None:
    """Provide only the three helpers imported by the fixed official file."""
    timm = types.ModuleType("timm")
    models = types.ModuleType("timm.models")
    layers = types.ModuleType("timm.models.layers")
    layers.DropPath = DropPath
    layers.to_2tuple = to_2tuple
    layers.trunc_normal_ = torch.nn.init.trunc_normal_
    timm.models = models
    models.layers = layers
    sys.modules.setdefault("timm", timm)
    sys.modules.setdefault("timm.models", models)
    sys.modules.setdefault("timm.models.layers", layers)


def _load_official_class(reference_path: Path):
    _install_timm_compatibility_shim()
    spec = importlib.util.spec_from_file_location(
        "fixed_official_network_swinir",
        reference_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load official source at {reference_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SwinIR


def build_models(reference_path: Path):
    official_class = _load_official_class(reference_path)
    torch.manual_seed(0)
    official = official_class(**SMALL_CONFIG)
    torch.manual_seed(0)
    independent = SwinIR(**SMALL_CONFIG)
    return official, independent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("references/network_swinir.py"),
    )
    args = parser.parse_args()
    if not args.reference.exists():
        raise SystemExit(
            f"missing {args.reference}; follow references/README.md to add it"
        )

    official, independent = build_models(args.reference)
    official_state = official.state_dict()
    independent_state = independent.state_dict()
    if official_state.keys() != independent_state.keys():
        missing = sorted(official_state.keys() - independent_state.keys())
        extra = sorted(independent_state.keys() - official_state.keys())
        raise SystemExit(f"state_dict mismatch: missing={missing}, extra={extra}")

    independent.load_state_dict(official_state, strict=True)
    official.eval()
    independent.eval()
    torch.manual_seed(1)
    x = torch.randn(1, 2, 8, 12)
    with torch.no_grad():
        expected = official(x)
        actual = independent(x)

    official_params = sum(parameter.numel() for parameter in official.parameters())
    independent_params = sum(
        parameter.numel() for parameter in independent.parameters()
    )
    max_error = (expected - actual).abs().max().item()
    print(f"official_parameters={official_params}")
    print(f"independent_parameters={independent_params}")
    print(f"max_abs_error={max_error:.9g}")
    if official_params != independent_params or max_error >= 1e-6:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
