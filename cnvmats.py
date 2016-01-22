# -*- coding: utf-8 -*-

import numpy as np


CIRC  = 'circ'
VALID = 'valid'
FULL  = 'full'


SHAPE_MISMATCH = 'shape mismatch'
MALFORMED_SHAPE = 'malformed shape'

def check_shape(actual, expected):
    if not np.all(actual == expected):
        msg = '%s, %s but expected %s' % (SHAPE_MISMATCH, str(actual), str(expected))
        raise ValueError(msg)

def shape_equal(shape1, shape2):
    def normalize(shape):
        if np.isscalar(shape): # scalars
            return shape
        elif isinstance(shape, np.ndarray) and shape.size == 1: # arrays
            return shape.flat[0]
        elif len(shape) == 1: # tuples and lists
            return shape[0]
        else: # anything else
            return shape
    return np.array_equal(normalize(shape1), normalize(shape2))


class Identity:
    """Represents the identity matrix.
    """

    def dot(self, g): return g

    @property
    def T(self): return self

I = Identity()


class PadMat:
    """Represents matrix that does padding or unpadding.
    """

    UP = 'up'
    DOWN = 'down'
    UNKNOWN_DIRECTION = 'unknown direction'
    UNIMPLEMENTED_NDIM = 'not implemented for ndarray.ndim=%d'

    def __init__(self, sfrom, sto, direction=UP):
        self.sfrom = sfrom
        self.sto = sto
        self.direction = direction

    def dot(self, g):
        check_shape(g.shape, self.sfrom)
        sfrom = np.array(self.sfrom)
        if np.all(sfrom > self.sto):
            return self.unpad(g, sfrom)
        elif np.all(sfrom < self.sto):
            return self.pad(g, sfrom)
        else:
            return g

    @property
    def T(self):
        return PadMat(self.sto, self.sfrom, self.direction)

    def pad(self, src, sfrom):
        pad_with = self.sto - sfrom
        if isinstance(pad_with, np.ndarray):
            pad_with = pad_with.flat[0]
        if self.direction == PadMat.UP:
            padding = (0, pad_with)
        elif self.direction == PadMat.DOWN:
            padding = (pad_with, 0)
        else:
            raise ValueError(UNKNOWN_DIRECTION)
        return np.pad(src, padding, mode='constant')

    def unpad(self, src, sfrom):
        if self.direction == PadMat.UP:
            if src.ndim == 1:
                sto = self.sto.flat[0] if isinstance(self.sto, np.ndarray) else self.sto
                return src[:sto]
            elif src.ndim == 2:
                return src[:self.sto[0], :self.sto[1]]
            else:
                raise ValueError(UNIMPLEMENTED_NDIM % src.ndim)
        elif self.direction == PadMat.DOWN:
            offset = sfrom - self.sto
            if src.ndim == 1:
                if isinstance(offset, np.ndarray):
                    offset = offset.flat[0]
                return src[offset:]
            elif src.ndim == 2:
                return src[offset[0]:, offset[1]:]
            else:
                raise ValueError(UNIMPLEMENTED_NDIM % src.ndim)
        else:
            raise ValueError(UNKNOWN_DIRECTION)


class CnvMat:
    """Represents a generic matrix that does convolution or correlation.
    """

    dtype = 'float64'
    UNSUPPORTED_ARGUMENT_TYPE = 'unsupported argument type'

    def __init__(self, f_freq, sg, g_pad=I, h_unpad=I, T=None):
        self.f_freq  = f_freq
        self.sg      = sg
        self.g_pad   = g_pad
        self.h_unpad = h_unpad
        self.T       = T or CnvMat(f_freq.conj(), self.sh, g_pad=h_unpad.T, h_unpad=g_pad.T, T=self)

    def toarray(self, dtype=None):
        array = np.zeros(self.shape, dtype or self.dtype)
        g = np.zeros(self.sg)
        for k in range(np.prod(g.shape)):
            g.flat[k] = 1
            array[:,k] = (self * g).flatten()
            g.flat[k] = 0
        return array

    @property
    def shape(self):
        return (np.prod(self.sh), np.prod(self.sg))

    def dot(self, rarg):
        if isinstance(rarg, CnvMat):
            rarg_array = rarg.toarray()
            return self.dot(rarg_array)
        elif isinstance(rarg, np.ndarray):
            if shape_equal(rarg.shape, self.sg):
                return self * rarg
            else:
                if rarg.shape[0] == np.prod(self.sg):
                    result = np.zeros((self.shape[0], rarg.shape[1]), rarg.dtype)
                    for i in range(result.shape[1]):
                        result[:,i] = (self * rarg[:,i].reshape(self.sg)).flatten()
                    return result
                else:
                    raise ValueError(SHAPE_MISMATCH)
        else:
            raise ValueError(UNSUPPORTED_ARGUMENT_TYPE)

    @property
    def sh(self):
        if isinstance(self.h_unpad, Identity):
            return self.f_freq.shape
        else:
            return self.h_unpad.sto

    def __mul__(self, g):
        check_shape(g.shape, self.sg)
        g_padded = self.g_pad.dot(g)
        g_freq = np.fft.fftn(g_padded)
        check_shape(g_freq.shape, self.f_freq.shape)
        h = np.fft.ifftn(g_freq * self.f_freq)
        h = self.h_unpad.dot(h)
        if self.dtype is None:
            return h
        else:
            if not self.dtype.startswith('complex'):
                h = h.real
            return h.astype(self.dtype)


def cnvmat(f_spat, sg, mode):
    """Constructs matrix that convolves `f_spat` with arguments from `sg` shape.
    """

    sg = sg if isinstance(sg, np.ndarray) else np.array(sg)
    sf = np.array(f_spat.shape)
    if not np.all(sg <= sf) and not np.all(sg >= sf):
        raise ValueError(SHAPE_MISMATCH)

    kwargs = {}
    if mode == CIRC:

        if np.all(sf <= sg): # role: A
            kwargs['f_pad'] = PadMat(sf, sg)
            kwargs['g_pad'] = I

        else: # role: X
            kwargs['f_pad'] = I
            kwargs['g_pad'] = PadMat(sg, sf)

    elif mode == VALID:

        if np.all(sf <= sg): #role A
            kwargs['f_pad']   = PadMat(sf, sg)
            kwargs['g_pad']   = I
            kwargs['h_unpad'] = PadMat(sg, sg - sf + 1, direction=PadMat.DOWN)

        else: # role: X
            kwargs['f_pad']   = I
            kwargs['g_pad']   = PadMat(sg, sf)
            kwargs['h_unpad'] = PadMat(sf, sf - sg + 1, direction=PadMat.DOWN)

    elif mode == FULL:
        sh = sf + sg - 1
        kwargs['f_pad'] = PadMat(sf, sh)
        kwargs['g_pad'] = PadMat(sg, sh)

    else:

        raise ValueError('unknown mode: "%s"' % mode)

    f_pad = kwargs.pop('f_pad')
    f_freq = np.fft.fftn(f_pad.dot(f_spat))
    mat = CnvMat(f_freq, sg, **kwargs)
    mat.mode = mode
    return mat


UNSUPPORTED_MODE = 'unsupported mode: %s'

def cnvmat_tp(f_spat, sh, mode):
    """Returns `F.T` where `F.mode == mode` and `F.sh == sh == F.T.sg`.
    """

    sf = np.array(f_spat.shape)
    sh = sh if isinstance(sh, np.ndarray) else np.array(sh)

    if mode == CIRC:

        # NOTE: we could implement this for `Ax` but not for `Xa`,
        # because this would require more information that we haven't available

        raise ValueError(UNSUPPORTED_MODE % mode)

    elif mode == VALID:

        if np.all(sh >= sf):
            sg = sf + sh - 1
        elif np.all(sh <= sf):
            sg = sf - sh + 1
        else:
            raise ValueError(SHAPE_MISMATCH)

    elif mode == FULL:

        if np.all(sh >= sf):
            sg = sh - sf + 1
        elif np.all(sh <= sf):
            sg = sf - sh + 1
        else:
            raise ValueError(SHAPE_MISMATCH)

    else:

        raise ValueError('unknown mode: "%s"' % mode)

    return cnvmat(f_spat, sg, mode).T

