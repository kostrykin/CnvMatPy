# -*- coding: utf-8 -*-

import cnvmats
import unittest
import numpy as np
import cv2
from matplotlib import pyplot as plt


def img_equals(actual, expected, tolerance=1e-3):
    return np.all(np.abs(actual - expected) <= tolerance)


class ImgCompTestCase(unittest.TestCase):

    def assertEqualImg(self, actual, expected, hint='', interp='none', tolerance=1e-8):
        ok = img_equals(actual, expected, tolerance)
        if not ok:
            plt.figure('Failure').suptitle(hint, fontsize=20)
            plt.subplot(1,3,1)
            plt.title('actual')
            plt.imshow(actual, 'gray', interpolation=interp)
            plt.colorbar()
            plt.subplot(1,3,2)
            plt.title('expected')
            plt.imshow(expected, 'gray', interpolation=interp)
            plt.colorbar()
            plt.subplot(1,3,3)
            plt.title('difference')
            plt.imshow(np.abs(actual-expected), 'gray', interpolation=interp)
            plt.colorbar()
            plt.show()
        self.assertTrue(ok, '%s failed' % hint)


class TestPadMat(unittest.TestCase):

    def test_sx_leq_sy(self):
        sx, sy = (2,2), (3,3)
        x = np.random.random(sx)
        y = cnvmats.PadMat(sx, sy).dot(x)
        for p in np.ndindex(y.shape):
            self.assertEqual(y[p], (x[p] if np.all(np.array(p) < np.array(sx)) else 0))

    def test_sx_eq_sy(self):
        sx, sy = (3,3), (3,3)
        x = np.random.random(sx)
        y = cnvmats.PadMat(sx, sy).dot(x)
        self.assertTrue(np.all(y == x))

    def test_sx_geq_sy(self):
        sx, sy = (4,4), (3,3)
        x = np.random.random(sx)
        y = cnvmats.PadMat(sx, sy).dot(x)
        self.assertTrue(np.all(y == x[:3, :3]))

    def test_sx_eq_sy(self):
        sx, sy = (3,3), (3,3)
        x = np.random.random(sx)
        y = cnvmats.PadMat(sx, sy).dot(x)
        self.assertTrue(np.all(y == x))


class TestCnvMat(ImgCompTestCase):

    def setUp(self):
        self.sa, self.sx = (3,3), (10,10)
        self.a = np.random.random(self.sa)
        self.x = np.round(255 * np.random.random(self.sx))
        self.modes = (cnvmats.CIRC, cnvmats.VALID, cnvmats.FULL)

    def test_XtpX(self):
        for mode in self.modes:
            X = cnvmats.cnvmat(self.x, self.sa, mode)
            X_array = X.toarray()
            XtpX = X.T.dot(X)
            self.assertEqual(XtpX.shape, (np.prod(X.sg), np.prod(X.sg)))
            self.assertAlmostEqual(np.linalg.norm(XtpX - X_array.T.dot(X_array)), 0)

    def test_Ax_equals_Xa(self):
        for mode in self.modes:
            X = cnvmats.cnvmat(self.x, self.sa, mode)
            A = cnvmats.cnvmat(self.a, self.sx, mode)
            self.assertEqual(np.linalg.norm(X * self.a - A * self.x), 0)

    def test_lena(self):
        sa = (30,30)
        x = cv2.imread('lena.png', 0)
        a = np.ones(sa) / np.prod(sa)
        sx = x.shape
        for mode in self.modes:
            X = cnvmats.cnvmat(x, sa, mode)
            A = cnvmats.cnvmat(a, sx, mode)
            Ax = A * x
            Xa = X * a
            expected = cv2.imread('lena-box30-%s.png' % mode, 0)
            self.assertEqualImg(Ax, expected, '$Ax$ %s' % mode, tolerance=1.1)
            self.assertEqualImg(Xa, expected, '$Xa$ %s' % mode, tolerance=1.1)

    def test_Xtp(self):
        for mode in self.modes:
            X = cnvmats.cnvmat(self.x, self.sa, mode)
            expected = X.toarray().T
            self.assertAlmostEqual(np.linalg.norm(X.T.toarray() - expected), 0)

    def test_Atp(self):
        for mode in self.modes:
            A = cnvmats.cnvmat(self.a, self.sx, mode)
            expected = A.toarray().T
            self.assertAlmostEqual(np.linalg.norm(A.T.toarray() - expected), 0)


if __name__ == '__main__':
    unittest.main()

