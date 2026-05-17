import unittest
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from brats_seg.metrics import dice_per_region, hd95_per_region


class MetricsTest(unittest.TestCase):
    def test_dice_per_region_perfect_overlap(self) -> None:
        truth = np.zeros((3, 4, 4), dtype=np.uint8)
        truth[:, 1:3, 1:3] = 1
        scores = dice_per_region(truth, truth)
        self.assertTrue(all(abs(value - 1.0) < 1e-6 for value in scores.values()))

    def test_hd95_zero_for_identical_masks(self) -> None:
        truth = np.zeros((3, 4, 4), dtype=np.uint8)
        truth[:, 1:3, 1:3] = 1
        scores = hd95_per_region(truth, truth)
        self.assertTrue(all(abs(value) < 1e-6 for value in scores.values()))


if __name__ == "__main__":
    unittest.main()
