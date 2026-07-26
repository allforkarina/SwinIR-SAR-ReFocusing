import pytest
import torch

from swinir.window_attention import WindowAttention


def test_relative_position_state_shapes_and_range():
    attention = WindowAttention(dim=24, window_size=(4, 4), num_heads=3)
    assert attention.relative_position_bias_table.shape == (49, 3)
    assert attention.relative_position_index.shape == (16, 16)
    assert attention.relative_position_index.min().item() == 0
    assert attention.relative_position_index.max().item() == 48


@pytest.mark.parametrize("use_mask", [False, True])
def test_attention_forward_backward(use_mask):
    attention = WindowAttention(dim=24, window_size=(4, 4), num_heads=3)
    x = torch.randn(4, 16, 24, requires_grad=True)
    mask = None
    if use_mask:
        mask = torch.zeros(2, 16, 16)
        mask[:, :8, 8:] = -100.0
        mask[:, 8:, :8] = -100.0

    output = attention(x, mask)
    output.square().mean().backward()

    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_attention_rejects_wrong_token_count():
    attention = WindowAttention(dim=24, window_size=(4, 4), num_heads=3)
    with pytest.raises(ValueError, match="tokens per window"):
        attention(torch.randn(1, 15, 24))
