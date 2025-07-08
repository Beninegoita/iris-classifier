"""
Very simple smoke test - verifies that training runs end-to-end and
achieves at least 90 % accuracy (Iris is an easy dataset).
Run with:  pytest
"""
from pathlib import Path
import sys

# Allow "src" to be importable when invoking pytest from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train import train  # noqa: E402  (import after path tweak)


def test_accuracy_threshold() -> None:
    acc = train(test_size=0.2, random_state=42)
    assert acc >= 0.90, f"Expected ≥ 0.90 accuracy, got {acc:.2%}"