import numpy as np
import cnvmats
import sys
from matplotlib import pyplot as plt
from PIL import Image, ImageSequence

def progress(msg, progress_min, progress_max, progress_val):
    progress_relative = (progress_val - progress_min) / float(progress_max - progress_min)
    progress_percentage = 100 * progress_relative
    text = '%5.1f%% %s' % (progress_percentage, msg)
    sys.stdout.write(text)
    if progress_val == progress_max:
        sys.stdout.write('\n')
    else:
        sys.stdout.write('\b'*len(text))
        sys.stdout.flush()

class BlindDeconv:

    def __init__(self, sa, mode, iters_count=1, epsilon=1e-8):
        self.sa = sa
        self.mode = mode
        self.iters_count = iters_count
        self.epsilon = epsilon

    def step(self, x, y):
        sa = self.sa
        sx = x.shape
        a = np.ones(sa)
        for iter_idx in range(self.iters_count):
            A = cnvmats.cnvmat(a, sx, self.mode)
            X = cnvmats.cnvmat(x, sa, self.mode)
            a_mult_update = self.mult_update(X, X.tp(), y, a)
            x_mult_update = self.mult_update(A, A.tp(), y, x)
            x = np.multiply(x, x_mult_update)
            a = np.multiply(a, a_mult_update)
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
        return (x,a)

    def mult_update(self, F, Ftp, g, h0):
        return non_neg(np.divide( \
            (Ftp*g).real + self.epsilon, \
            (Ftp*(F*h0)).real + self.epsilon))

def non_neg(x):
    x[x<0] = 0
    return x

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('USAGE: %s <y>') % sys.argv[0]
    else:
        filename = sys.argv[1]
        print('Opening %s... ' % filename)
        im = Image.open(filename)
        y = []
        for frame in ImageSequence.Iterator(im):
            np_frame = np.array(frame.getdata(), np.uint16).reshape(im.size[1], im.size[0])
            y.append(np_frame)
        print('Read %d frames.' % len(y))
        bd = BlindDeconv((30,30), 'circ', iters_count=2)
        (x,a) = bd.batch(y[-1], y)
        plt.imshow(x[-1], 'gray')
        plt.colorbar()
        plt.show()
