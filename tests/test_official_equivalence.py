from pathlib import Path

import pytest
import torch

from scripts.compare_official import build_models


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_MODEL = PROJECT_ROOT / "references" / "network_swinir.py"


@pytest.mark.skipif(
    not REFERENCE_MODEL.exists(),
    reason="place the fixed official network_swinir.py in references/",
)
def test_official_small_model_numerical_equivalence():
    official, independent = build_models(REFERENCE_MODEL)
    independent.load_state_dict(official.state_dict(), strict=True)
    official.eval()
    independent.eval()

    torch.manual_seed(0)
    x = torch.randn(1, 2, 8, 12)
    with torch.no_grad():
        expected = official(x)
        actual = independent(x)
    assert (expected - actual).abs().max().item() < 1e-6
