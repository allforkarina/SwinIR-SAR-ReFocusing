from pathlib import Path

import torch
import yaml

from swinir.model import SwinIR


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_standard_model():
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "swinir_same_size.yaml").read_text(
            encoding="utf-8"
        )
    )
    return SwinIR(**config["model"])


def test_standard_model_structure():
    model = load_standard_model()
    assert len(model.layers) == 6
    assert [len(layer.residual_group.blocks) for layer in model.layers] == [6] * 6
    for layer in model.layers:
        assert [block.shift_size for block in layer.residual_group.blocks] == [
            0,
            4,
            0,
            4,
            0,
            4,
        ]
    assert model.conv_first.in_channels == 2
    assert model.conv_last.out_channels == 2
    assert isinstance(model.norm, torch.nn.LayerNorm)
    assert isinstance(model.conv_after_body, torch.nn.Conv2d)
    assert model.patch_embed.patch_size == (1, 1)
    assert model.upscale == 1
    assert model.upsampler == ""


def test_standard_model_parameter_count_is_stable():
    model = load_standard_model()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    assert parameter_count == 11_500_922
