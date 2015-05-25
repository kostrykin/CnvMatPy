import cnvmats
import numpy as np
from matplotlib import pyplot as plt

modes = ('valid', 'full', 'circ')
sa, sx = (3,3), (7,7)
np.random.seed(1)
a = np.random.random(sa)

plt.subplot(2, len(modes), 1+len(modes)/2)
plt.imshow(a, 'gray', interpolation='none')
plt.title('$a$')
plt.xticks(np.arange(0, sa[0], 1))
plt.yticks(np.arange(0, sa[0], 1))

for mode_idx in range(len(modes)):
    mode = modes[mode_idx]
    A = cnvmats.cnvmat(a, sx, mode)

    plt.subplot(2, len(modes), 1+len(modes)+mode_idx)
    plt.imshow(A.toarray().real, 'gray', interpolation='none')
    plt.title('$A_{%s}$' % mode)

plt.show()
