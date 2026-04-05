from __future__ import annotations

from unittest.mock import patch

from mini_lewm.utils.device import resolve_device


def test_resolve_device_auto_prefers_cpu_when_mps_unavailable() -> None:
    with patch("torch.backends.mps.is_available", return_value=False):
        assert resolve_device("auto").type == "cpu"


def test_resolve_device_uses_mps_when_available() -> None:
    with patch("torch.backends.mps.is_available", return_value=True):
        assert resolve_device("mps").type == "mps"
