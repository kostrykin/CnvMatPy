# -*- coding: utf-8 -*-

import cnvmats
import unittest
import numpy as np
import cv2
from matplotlib import pyplot as plt

def img_equals(actual, expected):
    return np.linalg.norm(actual - expected, ord='fro') < np.prod(actual.shape)

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
        
class TestCircMat(unittest.TestCase):
    
    def test_shapes_Ax(self):
        sa, sx = (3,3), (7,7)
        a = np.random.random(sa)
        A = cnvmats.CircMat(a, sx)
        self.assertEquals(A.sh, sx)
        Atp = A.tp()
        self.assertEquals(Atp.f_spat.shape, sa)
        self.assertEquals(Atp.sg, sx)
        self.assertEquals(Atp.sh, sx)
        self.assertEquals(Atp.tp(), A)
    
    def test_shapes_Xa(self):
        sa, sx = (3,3), (7,7)
        x = np.random.random(sx)
        X = cnvmats.CircMat(x, sa)
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
        A = cnvmats.CircMat(a, x.shape)
        y = (A*x).real
        y_expected = cv2.imread('lena-box20-circ.png', 0)
        self.assertTrue(img_equals(y, y_expected))
    
    def test_lena_Xa(self):
        sa = (30,30)
        x = cv2.imread('lena.png', 0)
        a = np.ones(sa) / np.prod(sa)
        X = cnvmats.CircMat(x, a.shape)
        y = (X*a).real
        y_expected = cv2.imread('lena-box20-circ.png', 0)
        self.assertTrue(img_equals(y, y_expected))

if __name__ == '__main__':
    unittest.main()
