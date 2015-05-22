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
        
class CnvMat:
    """Implements matrix that performs circular convolution."""
    
    def __init__(self, f_spat, sg, sh=None):
        self.f_spat = f_spat
        self.sg = sg
        self.sh = sh if sh is not None else max(f_spat.shape, sg)
        if np.all(f_spat.shape <= sg):
            self.f_freq = np.fft.fft2(pad(f_spat, sg))
        else:
            self.f_freq = np.fft.fft2(f_spat)
        
    def __mul__(self, g_spat):
        assert self.sg == g_spat.shape
        if np.all(self.f_spat.shape <= self.sg):
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
        return CnvMat(tp_f_spat, tp_sg, tp_sh)
    
    def __eq__(self, other):
        return isinstance(other, self.__class__) \
            and np.all(self.f_spat == other.f_spat) \
            and self.sg == other.sg \
            and self.sh == other.sh \
            and np.all(self.f_freq == other.f_freq)
    
    def __ne__(self, other):
        return not self.__eq__(other)