import numpy as np
import scipy
from nnFuncionCosto import feedForward
# if available import pylab (from matlibplot)
try:
    import matplotlib.pylab as plt
except ImportError:
    pass

def predecir(theta1, theta2, x):
    m = np.shape(x)[0]
    a1, a2, a3, z2, z3 = feedForward(theta1, theta2, x)
    argmx = np.argmax(a3, axis=1)
    return np.reshape(argmx, (m, 1)).flatten()
