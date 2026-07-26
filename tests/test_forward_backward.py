import pytest
import torch

from swinir.model import SwinIR


def small_model(**overrides):
    config = {
        "img_size": 8,
        "patch_size": 1,
        "in_chans": 2,
        "embed_dim": 12,
        "depths": [2, 2],
        "num_heads": [3, 3],
        "window_size": 4,
        "mlp_ratio": 2,
        "drop_path_rate": 0.0,
        "upscale": 1,
        "upsampler": "",
    }
    config.update(overrides)
    return SwinIR(**config)


@pytest.mark.parametrize("shape", [(1, 2, 8, 8), (1, 2, 7, 5), (1, 2, 2, 3)])
def test_same_size_forward_for_regular_dynamic_and_tiny_inputs(shape):
    model = small_model()
    x = torch.randn(*shape)
    output = model(x)
    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def test_full_model_backward_has_finite_gradients():
    model = small_model()
    x = torch.randn(1, 2, 7, 5, requires_grad=True)
    output = model(x)
    output.square().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    gradients = [parameter.grad for parameter in model.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.parametrize(
    ("upsampler", "upscale"),
    [("pixelshuffle", 2), ("pixelshuffledirect", 2), ("nearest+conv", 2)],
)
def test_compatibility_upsampling_branches(upsampler, upscale):
    model = small_model(upsampler=upsampler, upscale=upscale)
    x = torch.randn(1, 2, 8, 8)
    assert model(x).shape == (1, 2, 16, 16)
