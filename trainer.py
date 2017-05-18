import numpy as np
import sys
from scipy.optimize import fmin_cg
from calcWeights import calcWeights
from nnFuncionCosto import nnFuncionCosto, nnFuncionGradiente

def trainer(initial_nn_params,x, y, inputL_size, hiddenL_size, outputL_size):

    calcW = calcWeights(inputL_size, hiddenL_size, outputL_size) 

    _lambda = 0.01
    max_iterations = 50
    iterations_counter = dict(val=0)

    def show_progress(current_x):
        iterations_counter['val'] += 1
        progress = iterations_counter['val'] * 100 // max_iterations
	
        sys.stdout.write('\r[{0}{1}] {2}% - iter:{3}'.format(
            '=' * (progress // 5),
            ' ' * ((104 - progress) // 5),
            progress, iterations_counter['val']
        ))

    # Fmincg!
    nn_params = fmin_cg(
        nnFuncionCosto,
        x0=initial_nn_params,
        args=(inputL_size, hiddenL_size, outputL_size, x, y, _lambda),
	fprime=nnFuncionGradiente,
        maxiter=max_iterations,
        callback=show_progress
    )

    theta1, theta2 = calcW.unroll_thetas(nn_params, inputL_size, hiddenL_size, outputL_size)

    return theta1, theta2
