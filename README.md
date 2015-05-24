# CnvMatPy

This repository contains the `cnvmats` Python 2.7 module. It provides a few classes, whose instances represent convolution and/or correlation matrices, like `A` and `X` in `Ax = Xa = a*x`, respectively. The classes of `A` and `X` support transposition using their `tp` function, besides of multiplication with numpy arrays. The convolution is implemented in frequency domain.

These objects are instantiated by the `cnvmats.cnvmat` function. This function has a `mode` argument that can be set to either `valid`, `full` or `circ`. The `cnvmats_test.py` file contains tests and examples.

![modes](https://github.com/kostrykin/CnvMatPy/blob/master/cnvmats_show.png?raw=true "modes")
