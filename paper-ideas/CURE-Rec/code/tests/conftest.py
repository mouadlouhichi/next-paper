from __future__ import annotations

from pathlib import Path

import pytest

from cure_rec.config import load_settings


@pytest.fixture
def settings():
    root = Path(__file__).resolve().parents[1]
    return load_settings(root / "configs" / "curesim_quickstart.yaml")
