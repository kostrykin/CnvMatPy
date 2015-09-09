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

class TestPad(unittest.TestCase):

    def test_sx_leq_sy(self):
        sx, sy = (2,2), (3,3)
        x = np.random.random(sx)
        y = cnvmats.pad(x, sy)
        for p in np.ndindex(y.shape):
            self.assertEqual(y[p], (x[p] if np.all(np.array(p) < np.array(sx)) else 0))
            
    def test_sx_eq_sy(self):
        sx, sy = (3,3), (3,3)
        x = np.random.random(sx)
        y = cnvmats.pad(x, sy)
        self.assertTrue(np.all(y == x))
            
    def test_sx_geq_sy(self):
        sx, sy = (3,3), (2,2)
        x = np.random.random(sx)
        with self.assertRaises(AssertionError):
            cnvmats.pad(x, sy)
            
    def test_sx_mixed_sy(self):
        sx, sy = (3,2), (2,3)
        x = np.random.random(sx)
        with self.assertRaises(AssertionError):
            cnvmats.pad(x, sy)

class TestUnpad(unittest.TestCase):

    def test_sx_geq_sy(self):
        sx, sy = (4,4), (3,3)
        x = np.random.random(sx)
        y = cnvmats.unpad(x, sy)
        self.assertTrue(np.all(y == x[:3, :3]))
    
    def test_sx_leq_sy(self):
        sx, sy = (3,3), (4,4)
        x = np.random.random(sx)
        with self.assertRaises(AssertionError):
            cnvmats.unpad(x, sy)
    
    def test_sx_eq_sy(self):
        sx, sy = (3,3), (3,3)
        x = np.random.random(sx)
        y = cnvmats.unpad(x, sy)
        self.assertTrue(np.all(y == x))
            
    def test_sx_mixed_sy(self):
        sx, sy = (3,4), (4,3)
        x = np.random.random(sx)
        with self.assertRaises(AssertionError):
            cnvmats.unpad(x, sy)

class TestFlip(unittest.TestCase):
    
    def test(self):
        sx = (3,3)
        x = np.zeros(sx)
        for p in np.ndindex(sx):
            x[p] = p[0] + p[1] * sx[0]
        y = cnvmats.flip(x)
        z = np.zeros(sx)
        for p in np.ndindex(sx):
            z[p] = (sx[0] - p[0] - 1) + (sx[1] - p[1] - 1) * sx[0]
        self.assertTrue(np.all(y == z))
        self.assertTrue(np.all(x == cnvmats.flip(y)))

class TestCircShift(unittest.TestCase):
    
    def test(self):
        sx = (3,3)
        x = np.zeros(sx)
        for p in np.ndindex(sx):
            x[p] = p[0] + p[1] * sx[0]
        y = cnvmats.circshift(x, +1)
        z = np.zeros(sx)
        for p in np.ndindex(sx):
            z[p] = (p[0] - 1) % sx[0] + ((p[1] - 1) % sx[1]) * sx[0]
        self.assertTrue(np.all(y == z))
        self.assertTrue(np.all(x == cnvmats.circshift(y, -1)))
        
class TestCircMat(ImgCompTestCase):
    
    def test_shapes_Ax(self):
        sa, sx = (3,3), (7,7)
        a = np.random.random(sa)
        A = cnvmats.cnvmat(a, sx, 'circ')
        self.assertEquals(A.sh, sx)
        Atp = A.tp()
        self.assertEquals(Atp.sg, sx)
        self.assertEquals(Atp.sh, sx)
    
    def test_shapes_Xa(self):
        sa, sx = (3,3), (7,7)
        x = np.random.random(sx)
        X = cnvmats.cnvmat(x, sa, 'circ')
        self.assertEquals(X.sh, sx)
        Xtp = X.tp()
        self.assertEquals(Xtp.f_spat.shape, sx)
        self.assertEquals(Xtp.sg, sx)
        self.assertEquals(Xtp.sh, sa)
        self.assertEquals(Xtp.tp(), X)
    
    def test_lena_Ax(self):
        sa = (30,30)
        x = cv2.imread('lena.png', 0)
        a = np.ones(sa) / np.prod(sa)
        A = cnvmats.cnvmat(a, x.shape, 'circ')
        y = (A*x).real
        y_expected = cv2.imread('lena-box30-circ.png', 0)
        self.assertEqualImg(y, y_expected, '$Ax$ circ', tolerance=1.1)
    
    def test_lena_Xa(self):
        sa = (30,30)
        x = cv2.imread('lena.png', 0)
        a = np.ones(sa) / np.prod(sa)
        X = cnvmats.cnvmat(x, a.shape, 'circ')
        y = (X*a).real
        y_expected = cv2.imread('lena-box30-circ.png', 0)
        self.assertEqualImg(y, y_expected, '$Xa$ circ', tolerance=1.1)
        
class TestValidMat(ImgCompTestCase):
    
    def test_shapes_Ax(self):
        sa, sx, sy = (3,3), (7,7), (5,5)
        a = np.random.random(sa)
        A = cnvmats.cnvmat(a, sx, 'valid')
        self.assertEquals(A.sh, sy)
        Atp = A.tp()
        self.assertEquals(Atp.circ.f_spat.shape, sa)
        self.assertEquals(Atp.sg, sy)
        self.assertEquals(Atp.sh, sx)
        self.assertEquals(Atp.tp(), A)
    
    def test_shapes_Xa(self):
        sa, sx, sy = (3,3), (7,7), (5,5)
        x = np.random.random(sx)
        X = cnvmats.cnvmat(x, sa, 'valid')
        self.assertEquals(X.sh, sy)
        Xtp = X.tp()
        self.assertEquals(Xtp.circ.f_spat.shape, sx)
        self.assertEquals(Xtp.sg, sy)
        self.assertEquals(Xtp.sh, sa)
        self.assertEquals(Xtp.tp(), X)
    
    def test_lena_Ax(self):
        sa = (30,30)
        x = cv2.imread('lena.png', 0)
        a = np.ones(sa) / np.prod(sa)
        A = cnvmats.cnvmat(a, x.shape, 'valid')
        y = (A*x).real
        y_expected = cv2.imread('lena-box30-valid.png', 0)
        self.assertEqualImg(y, y_expected, '$Ax$ valid', tolerance=1.1)
    
    def test_lena_Xa(self):
        sa = (30,30)
        x = cv2.imread('lena.png', 0)
        a = np.ones(sa) / np.prod(sa)
        X = cnvmats.cnvmat(x, a.shape, 'valid')
        y = (X*a).real
        y_expected = cv2.imread('lena-box30-valid.png', 0)
        self.assertEqualImg(y, y_expected, '$Xa$ valid', tolerance=1.1)
        
class TestFullMat(ImgCompTestCase):
    
    def test_shapes_Ax(self):
        sa, sx, sy = (3,3), (7,7), (9,9)
        a = np.random.random(sa)
        A = cnvmats.cnvmat(a, sx, 'full')
        self.assertEquals(A.sh, sy)
        Atp = A.tp()
        self.assertEquals(Atp.circ.f_spat.shape, sa)
        self.assertEquals(Atp.sg, sy)
        self.assertEquals(Atp.sh, sx)
        self.assertEquals(Atp.tp(), A)
    
    def test_shapes_Xa(self):
        sa, sx, sy = (3,3), (7,7), (9,9)
        x = np.random.random(sx)
        X = cnvmats.cnvmat(x, sa, 'full')
        self.assertEquals(X.sh, sy)
        Xtp = X.tp()
        self.assertEquals(Xtp.sg, sy)
        self.assertEquals(Xtp.sh, sa)
        self.assertEquals(Xtp.tp(), X)
    
    def test_lena_Ax(self):
        sa = (30,30)
        x = cv2.imread('lena.png', 0)
        a = np.ones(sa) / np.prod(sa)
        A = cnvmats.cnvmat(a, x.shape, 'full')
        y = (A*x).real
        y_expected = cv2.imread('lena-box30-full.png', 0)
        self.assertEqualImg(y, y_expected, '$Ax$ full', tolerance=1.1)
    
    def test_lena_Xa(self):
        sa = (30,30)
        x = cv2.imread('lena.png', 0)
        a = np.ones(sa) / np.prod(sa)
        X = cnvmats.cnvmat(x, a.shape, 'full')
        y = (X*a).real
        y_expected = cv2.imread('lena-box30-full.png', 0)
        self.assertEqualImg(y, y_expected, '$Xa$ full', tolerance=1.1)

class TestCnvMat(ImgCompTestCase):

    def setUp(self):
        self.sa, self.sx = (3,3), (10,10)
        self.a = np.random.random(self.sa)
        self.x = np.round(255 * np.random.random(self.sx))
        self.modes = ('valid', 'full', 'circ')

    def test_Ax_against_toarray(self):
        a, x, sa, sx = self.a, self.x, self.sa, self.sx
        for mode in self.modes:
            A = cnvmats.cnvmat(a, sx, mode)
            sy = A.sh
            Ax_actual = (A*x).real
            Ax_expected = A.toarray().dot(x.flatten()).reshape(sy).real
            y = Ax_actual
            Atpy_actual = (A.tp()*y).real
            Atpy_expected = A.toarray().T.dot(y.flatten()).reshape(sx).real
            self.assertEqualImg(Ax_actual, Ax_expected, '$Ax$ %s' % mode)
            self.assertEqualImg(A.tp().toarray().real, A.toarray().T.real, '$A^T$ %s' % mode)
            self.assertEqualImg(Atpy_actual, Atpy_expected, '$A^Ty$ %s' % mode)
            self.assertEqualImg(A.tp().tp().toarray().real, A.toarray().real, '$A^{TT}$ %s' % mode, tolerance=0)
    
    def test_Xa_against_toarray(self):
        a, x, sa, sx = self.a, self.x, self.sa, self.sx
        for mode in self.modes:
            X = cnvmats.cnvmat(x, sa, mode)
            sy = X.sh
            Xa_actual = (X*a).real
            Xa_expected = X.toarray().dot(a.flatten()).reshape(sy).real
            y = Xa_actual
            Xtpy_actual = (X.tp()*y).real
            Xtpy_expected = X.toarray().T.dot(y.flatten()).reshape(sa).real
            self.assertEqualImg(Xa_actual, Xa_expected, '$Xa$ %s' % mode)
            self.assertEqualImg(X.tp().toarray().real, X.toarray().T.real, '$X^T$ %s' % mode)
            self.assertEqualImg(Xtpy_actual, Xtpy_expected, '$X^Ty$ %s' % mode)
            self.assertEqualImg(X.tp().tp().toarray().real, X.toarray().real, '$X^{TT}$ %s' % mode, tolerance=0)

class TestCnvmatsTp(unittest.TestCase):
    
    def test_circ_Ax(self):
        sa, sx = (3,3), (7,7)
        a = np.random.random(sa)
        Atp = cnvmats.cnvmat(a, sx, 'circ').tp()
        Atp2 = cnvmats.cnvmat_tp(a, Atp.sg, 'circ')
        self.assertEquals(Atp, Atp2)
    
    def test_valid_Ax(self):
        sa, sx = (3,3), (7,7)
        a = np.random.random(sa)
        Atp = cnvmats.cnvmat(a, sx, 'valid').tp()
        Atp2 = cnvmats.cnvmat_tp(a, Atp.sg, 'valid')
        self.assertEquals(Atp, Atp2)
    
    def test_valid_Xa(self):
        sa, sx = (3,3), (7,7)
        x = np.random.random(sx)
        Xtp = cnvmats.cnvmat(x, sa, 'valid').tp()
        Xtp2 = cnvmats.cnvmat_tp(x, Xtp.sg, 'valid')
        self.assertEquals(Xtp, Xtp2)
    
    def test_full_Ax(self):
        sa, sx = (3,3), (7,7)
        a = np.random.random(sa)
        Atp = cnvmats.cnvmat(a, sx, 'full').tp()
        Atp2 = cnvmats.cnvmat_tp(a, Atp.sg, 'full')
        self.assertEquals(Atp, Atp2)
    
    def test_full_Xa(self):
        sa, sx = (3,3), (7,7)
        x = np.random.random(sx)
        Xtp = cnvmats.cnvmat(x, sa, 'full').tp()
        Xtp2 = cnvmats.cnvmat_tp(x, Xtp.sg, 'full')
        self.assertEquals(Xtp, Xtp2)

if __name__ == '__main__':
    unittest.main()
