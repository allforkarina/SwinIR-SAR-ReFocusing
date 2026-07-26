import torch

from swinir.basic_layer import BasicLayer
from swinir.swin_block import SwinTransformerBlock


def test_regular_window_block_forward():
    block = SwinTransformerBlock(
        dim=24,
        input_resolution=(8, 8),
        num_heads=3,
        window_size=4,
        shift_size=0,
        mlp_ratio=2,
    )
    x = torch.randn(2, 64, 24)
    assert block(x, (8, 8)).shape == x.shape


def test_shifted_window_mask_and_dynamic_forward():
    block = SwinTransformerBlock(
        dim=24,
        input_resolution=(8, 8),
        num_heads=3,
        window_size=4,
        shift_size=2,
        mlp_ratio=2,
    )
    assert set(block.attn_mask.unique().tolist()) == {-100.0, 0.0}
    x = torch.randn(1, 8 * 12, 24, requires_grad=True)
    output = block(x, (8, 12))
    output.mean().backward()
    assert output.shape == x.shape
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_basic_layer_alternates_window_shifts():
    layer = BasicLayer(
        dim=24,
        input_resolution=(8, 8),
        depth=6,
        num_heads=3,
        window_size=4,
        mlp_ratio=2,
    )
    assert [block.shift_size for block in layer.blocks] == [0, 2, 0, 2, 0, 2]
    x = torch.randn(1, 64, 24)
    assert layer(x, (8, 8)).shape == x.shape


def test_basic_layer_gradient_checkpointing_backward():
    layer = BasicLayer(
        dim=12,
        input_resolution=(8, 8),
        depth=2,
        num_heads=3,
        window_size=4,
        mlp_ratio=2,
        use_checkpoint=True,
    ).train()
    x = torch.randn(1, 64, 12, requires_grad=True)
    layer(x, (8, 8)).mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
