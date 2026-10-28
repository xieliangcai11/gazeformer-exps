import unittest
import numpy as np

from scripts.calibration.robust_calib import fit_ridge, fit_huber
from scripts.calibration.cc_eb import select_coverage


class CalibrationTests(unittest.TestCase):
    def test_ridge_recovers_affine(self):
        rng = np.random.RandomState(0)
        p = rng.rand(200)
        u = 1.2 * p - 0.1
        th = fit_ridge(p, u, 1e-6, 0.0)
        self.assertAlmostEqual(th[0], 1.2, places=3)
        self.assertAlmostEqual(th[1], -0.1, places=3)

    def test_huber_downweights_outlier(self):
        rng = np.random.RandomState(0)
        p = rng.rand(100)
        u = 1.0 * p + 0.0
        u[:2] += 5.0
        th_ridge = fit_ridge(p, u, 1e-3, 0.0)
        th_huber, w = fit_huber(p, u, 1e-3, 0.0, delta=1.345)
        self.assertLess(abs(th_huber[0] - 1.0), abs(th_ridge[0] - 1.0))
        self.assertTrue((w[:2] < 0.5).all())

    def test_coverage_selection_k_unique(self):
        pred = np.random.RandomState(1).rand(100, 2)
        rng = np.random.RandomState(0)
        sel = select_coverage(pred, 9, rng)
        self.assertEqual(len(sel), len(set(sel)))
        self.assertEqual(len(sel), 9)
        self.assertTrue(all(0 <= i < len(pred) for i in sel))


if __name__ == "__main__":
    unittest.main()
