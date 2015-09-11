# -*- coding: utf-8 -*-

import numpy as np

def pad(x, sy, offset=None, val=0):
    """Returns copy of `x` that is `val`-padded to `sy` size."""
    
    sy = np.array(sy)
    offset = np.array(offset) if offset is not None else np.zeros(sy.size)
    assert x.ndim == np.size(sy), 'ndim mismatch: %d == %d' % (x.ndim, np.size(sy))
    assert np.all(np.array(x.shape) <= sy), 'invalid shapes: %s <= %s' % (str(x.shape), str(sy))
    assert x.ndim == offset.size, 'ndim mismatch: %d == %d' % (x.ndim, offset.size)
    y = np.zeros(sy, x.dtype)
    if val != 0:
        y.fill(val)
    p0, p1 = offset, offset+np.array(x.shape)
    if x.ndim == 1:
        y[p0[0]:p1[0]] = x
    elif x.ndim == 2:
        y[p0[0]:p1[0], p0[1]:p1[1]] = x
    else:
        raise ValueError('ndim must be 1 or 2')
    return y
    
def unpad(x, sy, offset=None):
    """Returns copy of `x` that is cropped to `sx` size.
    """
    sy = np.array(sy)
    offset = np.array(offset) if offset is not None else np.zeros(sy.size)
    assert x.ndim == sy.size, 'ndim mismatch: %d == %d' % (x.ndim, sy.size)
    assert np.all(np.array(x.shape) >= sy), 'invalid shapes: %s <= %s' % (str(x.shape), str(sy))
    assert x.ndim == offset.size, 'ndim mismatch: %d == %d' % (x.ndim, offset.size)
    p0, p1 = offset, offset+sy
    if x.ndim == 1:
        return x[p0[0]:p1[0]]
    elif x.ndim == 2:
        return x[p0[0]:p1[0], p0[1]:p1[1]]
    else:
        raise ValueError('ndim must be 1 or 2')
    
def flip(x):
    """Returns copy of `x` flipped along each axis.
    """ 
    x_flipped_ud = np.flipud(x)
    if x.ndim == 1:
        return x_flipped_ud
    elif x.ndim == 2:
        return np.fliplr(x_flipped_ud)
    else:
        raise ValueError('ndim must be 1 or 2')

def circshift(x, shift):
    """Rolls `x` by `shift` along each axis.
    """
    if isinstance(shift, tuple) or isinstance(shift, list):
        assert x.ndim == len(shift), 'ndim mismatch: %d == %d' % (x.ndim, len(shift))
    else:
        shift = [shift]*x.ndim
    for d in range(x.ndim):
        x = np.roll(x, shift[d], axis=d)
    return x

def cnvmat(f_spat, sg, mode):
    if mode == 'circ':
        return CircMat(f_spat, sg)
    elif mode == 'valid':
        return ValidMat(f_spat, sg)
    elif mode == 'full':
        return FullMat(f_spat, sg)
    else:
        raise ValueError('unknown mode: "%s"' % mode)

def cnvmat_tp(f_spat, sh, mode):
    """Returns `F.tp()` where `F.mode == mode` and `F.sh == sh == F.tp().sg`.
    """
    sf = np.array(f_spat.shape)
    sh = np.array(sh)
    if mode == 'circ':
        # NOTE: only `Ax` is supported here, but not `Xa`,
        # because this would require more information that we haven't available
        sg = tuple(sh)
    elif mode == 'valid':
        if np.all(sh >= sf):
            sg = tuple(sf + sh - 1)
        elif np.all(sh <= sf):
            sg = tuple(sf - sh + 1)
        else:
            raise ValueError('shape mismatch')
    elif mode == 'full':
        if np.all(sh >= sf):
            sg = tuple(sh - sf + 1)
        elif np.all(sh <= sf):
            sg = tuple(sf - sh + 1)
        else:
            raise ValueError('shape mismatch')
    else:
        raise ValueError('unknown mode: "%s"' % mode)
    return cnvmat(f_spat, sg, mode).tp()
        
def check_shape(actual, expected):
    if actual != expected:
        msg = 'shape mismatch, %s but expected %s' % (str(actual), str(expected))
        raise ValueError(msg)

class CnvMat:
    """Base class of objects that represent convolution/correlation matrices.
    """
    def toarray(self):
        array = np.zeros((np.prod(self.sh), np.prod(self.sg)), 'complex')
        g = np.zeros(self.sg)
        for k in range(np.prod(g.shape)):
            g.flat[k] = 1
            array[:,k] = (self * g).flatten()
            g.flat[k] = 0
        return array

class CircMat(CnvMat):
    """Represents matrix that performs *circular* convolution.
    """
    def __init__(self, f_spat, sg, sh=None):
        self.f_spat = f_spat
        self.sf = f_spat.shape
        self.sg = sg
        self.sh = sh if sh is not None else max(self.sf, sg)
        if np.all(np.array(self.sf) <= np.array(sg)):
            self.f_freq = np.fft.fft2(pad(f_spat, sg))
        elif np.all(np.array(self.sf) >= np.array(sg)):
            self.f_freq = np.fft.fft2(f_spat)
        else:
            raise ValueError('shape mismatch')
        
    def __mul__(self, g_spat):
        check_shape(g_spat.shape, self.sg)
        if self.sf <= self.sg:
            g_freq = np.fft.fft2(g_spat)
        else:
            g_freq = np.fft.fft2(pad(g_spat, self.sf))
        h_freq = np.multiply(self.f_freq, g_freq)
        h_spat = unpad(np.fft.ifft2(h_freq), self.sh)
        return h_spat
        
    def tp(self):
        f_spat_padded = self.f_spat if self.sf >= self.sg else pad(self.f_spat, self.sg)
        tp_f_spat = circshift(flip(f_spat_padded), +1)
        tp_sg = self.sh
        tp_sh = self.sg
        return CircMat(tp_f_spat, tp_sg, tp_sh)
    
    def __eq__(self, other):
        return isinstance(other, self.__class__) \
            and np.all(self.f_spat == other.f_spat) \
            and self.sg == other.sg \
            and self.sh == other.sh \
            and np.all(self.f_freq == other.f_freq)
    
    def __ne__(self, other):
        return not self.__eq__(other)

class ValidMat(CnvMat):
    """Represents matrix that performs *valid* convolution.
    """
    def __init__(self, f_spat, sg):
        self.circ = CircMat(f_spat, sg)
        self.sf = f_spat.shape
        self.sg = sg
        sf = np.array(self.sf)
        sg = np.array(sg)
        if np.all(sf <= sg):
            self.sh = tuple(sg-sf+1)
        elif np.all(sf >= sg):
            self.sh = tuple(sf-sg+1)
        else:
            raise ValueError('shape mismatch')

    def __mul__(self, g_spat):
        h_circ = self.circ * g_spat
        offset = np.subtract(h_circ.shape, self.sh)
        h_valid = unpad(h_circ, self.sh, offset)
        return h_valid

    def tp(self):
        tp_f_spat = flip(self.circ.f_spat)
        if self.sf <= self.sg:
            return FullMat(tp_f_spat, self.sh)
        else:
            return ValidMat(tp_f_spat, self.sh)

    def __eq__(self, other):
        return isinstance(other, self.__class__) \
            and self.circ == other.circ \
            and self.sf == other.sf \
            and self.sg == other.sg \
            and self.sh == other.sh

    def __ne__(self, other):
        return not self.__eq__(other)

class FullMat(CnvMat):
    """Represents matrix that performs *full* convolution.
    """
    def __init__(self, f_spat, sg):
        self.sf = f_spat.shape
        self.sg = sg
        sf = np.array(self.sf)
        sg = np.array(sg)
        self.sh = tuple(sf+sg-1)
        if np.all(sf <= sg):
            circ_sg = self.sh
        elif np.all(sf >= sg):
            f_spat = pad(f_spat, self.sh)
            circ_sg = self.sg
        else:
            raise ValueError('shape mismatch')
        self.circ = CircMat(f_spat, circ_sg)

    def __mul__(self, g_spat):
        check_shape(g_spat.shape, self.sg)
        if self.sg != self.circ.sg:
            g_spat = pad(g_spat, self.circ.sg)
        return self.circ * g_spat

    def tp(self):
        tp_f_spat = flip(unpad(self.circ.f_spat, self.sf))
        return ValidMat(tp_f_spat, self.sh)

    def __eq__(self, other):
        return isinstance(other, self.__class__) \
            and self.circ == other.circ \
            and self.sf == other.sf \
            and self.sg == other.sg \
            and self.sh == other.sh

    def __ne__(self, other):
        return not self.__eq__(other)
