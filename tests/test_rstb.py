import pytest
import torch

from swinir.patch_ops import PatchEmbed, PatchUnEmbed
from swinir.rstb import RSTB


@pytest.mark.parametrize("use_norm", [False, True])
def test_patch_embed_unembed_shape_and_inverse_without_norm(use_norm):
    norm_layer = torch.nn.LayerNorm if use_norm else None
    embed = PatchEmbed(
        img_size=8,
        patch_size=1,
        in_chans=12,
        embed_dim=12,
        norm_layer=norm_layer,
    )
    unembed = PatchUnEmbed(
        img_size=8,
        patch_size=1,
        in_chans=12,
        embed_dim=12,
    )
    x = torch.randn(2, 12, 8, 6)
    tokens = embed(x)
    restored = unembed(tokens, (8, 6))
    assert restored.shape == x.shape
    if not use_norm:
        assert torch.equal(restored, x)


@pytest.mark.parametrize("resi_connection", ["1conv", "3conv"])
def test_rstb_shape_and_gradient(resi_connection):
    block = RSTB(
        dim=12,
        input_resolution=(8, 8),
        depth=2,
        num_heads=3,
        window_size=4,
        mlp_ratio=2,
        img_size=8,
        patch_size=1,
        resi_connection=resi_connection,
    )
    x = torch.randn(1, 64, 12, requires_grad=True)
    output = block(x, (8, 8))
    output.mean().backward()
    assert output.shape == x.shape
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
