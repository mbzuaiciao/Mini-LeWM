from __future__ import annotations

import torch

from mini_lewm.losses.sigreg import sigreg_loss


def test_sigreg_returns_scalar() -> None:
    loss = sigreg_loss(torch.randn(8, 32))
    assert loss.ndim == 0


def test_sigreg_has_no_nan() -> None:
    loss = sigreg_loss(torch.randn(8, 32))
    assert not torch.isnan(loss)
