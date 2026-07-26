import pytest
import torch

from swinir.common import DropPath, to_2tuple
from swinir.window_ops import window_partition, window_reverse


def test_to_2tuple():
    assert to_2tuple(8) == (8, 8)
    assert to_2tuple((4, 8)) == (4, 8)


def test_window_partition_and_reverse_are_exactly_invertible():
    x = torch.randn(2, 16, 24, 8)
    windows = window_partition(x, 8)
    restored = window_reverse(windows, 8, 16, 24)
    assert windows.shape == (12, 8, 8, 8)
    assert torch.equal(restored, x)


def test_window_partition_rejects_non_divisible_shape():
    with pytest.raises(ValueError, match="must be divisible"):
        window_partition(torch.randn(1, 15, 16, 3), 8)


def test_drop_path_eval_and_zero_probability_are_identity():
    x = torch.randn(4, 5, 6)
    layer = DropPath(0.5).eval()
    assert torch.equal(layer(x), x)
    assert torch.equal(DropPath(0.0).train()(x), x)


def test_drop_path_training_preserves_shape_and_finiteness():
    torch.manual_seed(0)
    output = DropPath(0.25).train()(torch.ones(128, 3, 4))
    assert output.shape == (128, 3, 4)
    assert torch.isfinite(output).all()
