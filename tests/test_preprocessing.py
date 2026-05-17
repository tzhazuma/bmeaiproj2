import unittest
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from brats_seg.preprocessing import compute_foreground_bbox, segmentation_to_regions, zscore_normalize


class PreprocessingTest(unittest.TestCase):
    def test_zscore_normalize_preserves_background(self) -> None:
        volume = np.array([[0, 0, 2], [0, 4, 6]], dtype=np.float32)
        normalized = zscore_normalize(volume)
        self.assertEqual(float(normalized[0, 0]), 0.0)
        self.assertAlmostEqual(float(normalized[volume != 0].mean()), 0.0, places=5)

    def test_segmentation_to_regions_supports_label_three(self) -> None:
        seg = np.array([[[0, 1], [2, 3]]], dtype=np.float32)
        regions = segmentation_to_regions(seg)
        self.assertEqual(regions.shape[0], 3)
        self.assertEqual(int(regions[2, 0, 1, 1]), 1)

    def test_compute_foreground_bbox(self) -> None:
        volumes = np.zeros((4, 5, 6, 7), dtype=np.float32)
        volumes[:, 1:4, 2:5, 3:6] = 1
        bbox = compute_foreground_bbox(volumes)
        self.assertEqual((bbox.z_min, bbox.z_max, bbox.y_min, bbox.y_max, bbox.x_min, bbox.x_max), (1, 4, 2, 5, 3, 6))


if __name__ == "__main__":
    unittest.main()
