import numpy as np
import cnvmats
import sys
import cv2
import math
from matplotlib import pyplot as plt

# based on
# ONLINE BLIND DECONVOLUTION FOR ASTRONOMICAL IMAGES, Harmeling et al., 2009

def progress(msg, progress_min, progress_max, progress_val):
    progress_relative = (progress_val-progress_min) / float(progress_max-progress_min)
    progress_percentage = 100 * progress_relative
    text = '%5.1f%% %s' % (progress_percentage, msg)
    sys.stdout.write(text)
    if progress_val == progress_max:
        sys.stdout.write('\n')
    else:
        sys.stdout.write('\b'*len(text))
        sys.stdout.flush()

class BlindDeconv:

    def __init__(self, sa, mode, itrs_count=1, epsilon=1e-8):
        self.sa = sa
        self.mode = mode
        self.itrs_count = itrs_count
        self.epsilon = epsilon

    def step(self, x, y):
        sa = self.sa
        sx = x.shape
        a = np.ones(sa)
        for iter_idx in range(self.itrs_count):
            A = cnvmats.cnvmat(a, sx, self.mode)
            X = cnvmats.cnvmat(x, sa, self.mode)
            a_mult_update = self.mult_update(X, X.tp(), y, a)
            x_mult_update = self.mult_update(A, A.tp(), y, x)
            x = x * x_mult_update
            a = a * a_mult_update
        return (x,a)

    def batch(self, x0, y, steps_count=None):
        steps_count = steps_count if steps_count is not None else len(y)
        x,a = [],[]
        x_last = x0
        for i in range(steps_count):
            y_i = y[i % len(y)]
            (x_i, a_i) = self.step(x_last, y_i)
            x.append(x_i)
            a.append(a_i)
            progress('blind deconvolution', 0, steps_count, i+1)
            x_last = x_i
        return (x,a)

    def mult_update(self, F, Ftp, g, h0):
        return non_neg(np.divide( \
            (Ftp*g).real + self.epsilon, \
            (Ftp*(F*h0)).real + self.epsilon))

def non_neg(x):
    x[x<0] = 0
    return x

def sq(x):
    return x*x

def blurmat(sa, sx, mode, gaussian=False):
    sigma2 = sq(np.divide(sa[0], 3))
    h_blur = np.ones(sa)
    if gaussian:
        p0 = np.divide(sa, 2)
        for p in np.ndindex(sa):
            p_dist = p - p0
            h_blur[p] = math.exp(-np.dot(p_dist, p_dist) / (2*sigma2))
    h_blur /= h_blur.sum()
    return cnvmats.cnvmat(h_blur, sx, mode)

def run(filename, y_count, steps_count, itrs_count, sa, noise_s2, mode):
    x_true = cv2.imread(filename, 0)
    y = [None]*y_count
    sx = x_true.shape
    for i in range(y_count):
        msg = 'creating %d input images from %s' % (y_count, filename)
        progress(msg, 0, y_count-1, i)
        a = np.random.random(sa)
        a = a / a.sum()
        A = cnvmats.cnvmat(a, sx, mode)
        sy = A.sh
        y[i] = (A * x_true).real + noise_s2 * np.random.randn(*sy)
    x0 = (blurmat(sa, sx, mode).tp() * np.mean(np.dstack(y), axis=2)).real
    bd = BlindDeconv(sa, mode, itrs_count=itrs_count)
    (x,a) = bd.batch(x0, y, steps_count=steps_count)
    rms = [np.sqrt(sq(x_i-x_true).sum() / np.prod(x_i.shape)) for x_i in x]
    plt.figure(figsize=(10,3))
    plt.subplot(1,3,1)
    plt.imshow(x0, 'gray')
    plt.title('$x_0$')
    plt.subplot(1,3,2)
    plt.imshow(x[-1], 'gray')
    plt.clim(0, 255)
    plt.colorbar(use_gridspec=True)
    plt.title('$x_{%d}$' % len(x))
    plt.subplot(1,3,3)
    plt.semilogx(np.array(range(len(rms)))+1, rms)
    plt.xlabel('iteration')
    plt.title('RMS')
    plt.grid()
    plt.tight_layout()
    plt.show()

run('lena.png', 33, 100, 2, (30,30), 10, 'valid')

