# -*- coding: utf-8 -*-

import numpy as np

def pad(x, sy):
    """Returns copy of 'x' that is zero-padded to 'sy' size."""
    
    sy = np.array(sy)
    assert x.ndim == np.size(sy), 'ndim mismatch: %d == %d' % (x.ndim, np.size(sy))
    assert np.all(np.array(x.shape) <= sy), 'invalid shapes: %s <= %s' % (str(x.shape), str(sy))
    y = np.zeros(sy)
    if x.ndim == 1:
        y[:x.shape[0]] = x
    elif x.ndim == 2:
        y[:x.shape[0], :x.shape[1]] = x
    else:
        raise ValueError('ndim must be 1 or 2')
    return y
    
def unpad(x, sy):
    """Returns copy of 'x' that is cropped to 'sx' size."""
    
    sy = np.array(sy)
    assert x.ndim == np.size(sy), 'ndim mismatch: %d == %d' % (x.ndim, np.size(sy))
    assert np.all(np.array(x.shape) >= sy), 'invalid shapes: %s <= %s' % (str(x.shape), str(sy))
    if x.ndim == 1:
        return x[:sy[0]]
    elif x.ndim == 2:
        return x[:sy[0], :sy[1]]
    else:
        raise ValueError('ndim must be 1 or 2')
    
def flip(x):
    """Returns copy of 'x' flipped along each axis."""
    
    x_flipped_ud = np.flipud(x)
    if x.ndim == 1:
        return x_flipped_ud
    elif x.ndim == 2:
        return np.fliplr(x_flipped_ud)
    else:
        raise ValueError('ndim must be 1 or 2')

def cnvmat(f_spat, sg, mode):
    if mode == 'circ':
        return CircMat(f_spat, sg)
    elif mode == 'valid':
        return ValidMat(f_spat, sg)
    elif mode == 'full':
        return FullMat(f_spat, sg)
    else:
        raise ValueError('unknown mode: "%s"' % mode)
        
class CircMat:
    """Represents matrix that performs circular convolution."""
    
    def __init__(self, f_spat, sg, sh=None):
        self.f_spat = f_spat
        self.sg = sg
        self.sh = sh if sh is not None else max(f_spat.shape, sg)
        if np.all(np.array(f_spat.shape) <= np.array(sg)):
            self.f_freq = np.fft.fft2(pad(f_spat, sg))
        elif np.all(np.array(f_spat.shape) >= np.array(sg)):
            self.f_freq = np.fft.fft2(f_spat)
        else:
            raise ValueError('shape mismatch')
        
    def __mul__(self, g_spat):
        assert self.sg == g_spat.shape, 'shape mismatch'
        if self.f_spat.shape <= self.sg:
            g_freq = np.fft.fft2(g_spat)
        else:
            g_freq = np.fft.fft2(pad(g_spat, self.f_spat.shape))
        h_freq = np.multiply(self.f_freq, g_freq)
        h_spat = unpad(np.fft.ifft2(h_freq), self.sh)
        return h_spat
        
    def tp(self):
        tp_f_spat = flip(self.f_spat)
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

class ValidMat:
    """Represents matrix that performs valid convolution."""

    def __init__(self, f_spat, sg):
        self.circ = CircMat(f_spat, sg)
        sf = np.array(f_spat.shape)
        sg = np.array(sg)
        if np.all(sf <= sg):
            self.sh = sf-sg+1
        elif np.all(sf <= sg):
            self.sh = sg-sf+1
        else:
            raise ValueError('shape mismatch')

    def __mul__(self, g_spat):
        h_circ = self.circ * g_spat
        h_valid = unpad(h_circ, sh)
        return h_valid

    def tp(self):
        tp_f_spat = flip(self.f_spat)
        tp_sg = self.sh
        if selff.circ.f_spat.shape <= self.circ.sg:
            return FullMat(tp_f_spat, tp_sg)
        else
            return ValidMat(tp_f_spat, tp_sg)

    def __eq__(self, other):
        return isinstance(other, self.__class__) \
            and self.circ == other.circ \
            and self.sh == other.sh

    def __ne__(self, other):
        return not self.__eq__(other)

class FullMat:
    """Represents matrix that performs full convolution."""

    def __init__(self, f_spat, sg):
        self.sg = sg
        sf = np.array(f_spat.shape)
        sg = np.array(sg)
        if np.all(sf <= sg):
            sg = sf + sg - 1
        elif np.all(sf >= sg):
            sh = sf + sg - 1
            f_spat = pad(f_spat, sh)
        else:
            raise ValueError('shape mismatch')
        self.circ = CircMat(f_spat, tuple(sg.tolist()))

    def __mul__(self, g_spat):
        assert self.sg == g_spat.shape, 'shape mismatch'
        if self.sg != self.circ.sg:
            g_spat = pad(g_spat, self.circ.sg)
        return self.circ * g_spat

    def tp(self):
        pass

    def __eq__(self, other):
        return isinstance(other, self.__class__) \
            and self.circ == other.circ \
            and self.sh == other.sh

    def __ne__(self, other):
        return not self.__eq__(other)
