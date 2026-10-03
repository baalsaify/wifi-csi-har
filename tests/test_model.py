import pytest
import torch

from csi_har.config import N_CLASSES
from csi_har.models import HARCNN


@pytest.mark.parametrize("length", [64, 256, 300])
def test_forward_shape_for_any_length(length):
    model = HARCNN().eval()
    with torch.no_grad():
        out = model(torch.randn(4, 90, length))
    assert out.shape == (4, N_CLASSES)


def test_model_is_small_enough_to_ship():
    params = sum(p.numel() for p in HARCNN().parameters())
    assert params < 1_000_000
