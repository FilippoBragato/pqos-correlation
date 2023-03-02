import numpy as np
from scipy.fft import fftn, ifftn

def comupte_correlation(x,y):
    cross_energy = np.sqrt((np.sum(x**2) * np.sum(y**2)))
    return np.sum(x*y)/cross_energy

def comupte_cross_correlation(a, b):
    ma = np.mean(a)
    mb = np.mean(b)
    sa = np.std(a)
    sb = np.std(b)

    a = (a - ma) / sa
    b = (b - mb) / sb

    fa = fftn(a)
    fb = fftn(b)

    fab = np.real(ifftn(fa * np.conj(fb)))

    return fab
