import numpy as np

def comupte_correlation(x,y):
    cross_energy = (np.sum(x.data**2) * np.sum(y.data**2))**0.5
    return np.sum(x.data*y.data)/cross_energy