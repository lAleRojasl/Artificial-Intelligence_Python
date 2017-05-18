import numpy as np
import scipy
from scipy.special import expit
import matplotlib.pylab as plt


def sigmoidGradient(z):
    g = np.zeros(np.shape(z))
    g = expit(z)*(1.-expit(z))
    return g
