import numpy as np


def _as_array(x):
    return np.asarray(x, dtype=float)


def gradient(func, x, eps=1e-6):
    x = _as_array(x)
    g = np.zeros_like(x)
    for i in range(x.size):
        xp = x.copy()
        xm = x.copy()
        xp[i] += eps
        xm[i] -= eps
        g[i] = (func(xp) - func(xm)) / (2.0 * eps)
    return g


def jacobian(func, x, eps=1e-6):
    x = _as_array(x)
    y0 = _as_array(func(x))
    j = np.zeros((y0.size, x.size))
    for i in range(x.size):
        xp = x.copy()
        xm = x.copy()
        xp[i] += eps
        xm[i] -= eps
        j[:, i] = (_as_array(func(xp)) - _as_array(func(xm))) / (2.0 * eps)
    return j


def hessian(func, x, eps=1e-5):
    x = _as_array(x)
    n = x.size
    h = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            xpp = x.copy()
            xpm = x.copy()
            xmp = x.copy()
            xmm = x.copy()
            xpp[i] += eps
            xpp[j] += eps
            xpm[i] += eps
            xpm[j] -= eps
            xmp[i] -= eps
            xmp[j] += eps
            xmm[i] -= eps
            xmm[j] -= eps
            h[i, j] = (
                func(xpp) - func(xpm) - func(xmp) + func(xmm)
            ) / (4.0 * eps * eps)
    return h
