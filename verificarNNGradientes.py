from calcWeights import calcWeights as calcW
from gradientChecking import gradientChecking
from nnFuncionCosto import nnFuncionCosto, nnFuncionGradiente
import numpy as np


def verificarNNGradientes(lmbda=0):

	input_layer_size = 3
	hidden_layer_size = 5
	num_labels = 3
	m = 5
	
	theta1 = calcW.debugInitWeights(hidden_layer_size, input_layer_size)
	theta2 = calcW.debugInitWeights(num_labels, hidden_layer_size)

	x = calcW.debugInitWeights(m, input_layer_size - 1)
	y = np.mod(np.arange(1, m+1), num_labels).T

	nn_params = np.concatenate((theta1.T.ravel(), theta2.T.ravel()))

	def costFunc(p):
		return nnFuncionCosto(p, input_layer_size, hidden_layer_size, num_labels,
				      x, y, lmbda)

   	def gradFunc(p):
		return nnFuncionGradiente(p, input_layer_size, hidden_layer_size, num_labels,
				         x, y, lmbda)

	grad = gradFunc(nn_params)
	numgrad = gradientChecking(costFunc, nn_params)

	print('Gradientes:')
	for i in range(len(grad)):
		print(numgrad[i],grad[i])

	diff = np.linalg.norm(numgrad-grad) / np.linalg.norm(numgrad+grad)
	
	print('\nDiferencia relativa (debe ser menor a 1e-9) : '+ str(diff))

