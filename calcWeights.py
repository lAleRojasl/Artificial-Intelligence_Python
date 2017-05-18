import scipy.io as sio
import numpy as np
import math

# Clase para manejar los pesos aleatorios (Theta1 y Theta2) 
# Sus dimensiones dependen de la cantidad de hidden layers ingresadas por el usuario,
# asi como de la cantidad de entradas y salidas.
# Ej: Si son N hidden layers, como tenemos 784 neuronas en la capa de entrada (imagenes 28x28)
#     entonces las dimensiones seran: Theta1 (Nx785) Theta2 (10x(N+1))

class calcWeights(object):

	# Inicializador de la clase, recibe:
	# inputL = cantidad de neuronas de entrada (en nuestro caso 784)
	# hiddenL = cantidad de hidden layers (definida por usuario)
	# outputL = cantidad de neuronas de salida (1 por posible numero = 10)
	# epsilon = define el rango para generar los pesos aleatorios
	def __init__(self, inputL, hiddenL, outputL):
		self.inputL_size = inputL
		self.hiddenL_size = hiddenL
		self.outputL_size = outputL

	# Funcion para generar un nuevo archivo Weights.mat con thetas aleatorios
	def newWeightsFile(self):
		theta1 = self.randInitWeights(self.inputL_size, self.hiddenL_size)
		print('Theta1 creado! Dimensiones: '+str(np.shape(theta1)))
		theta2 = self.randInitWeights(self.hiddenL_size, self.outputL_size)
		print('Theta2 creado! Dimensiones: '+str(np.shape(theta2)))
		sio.savemat('Weights.mat',{'Theta1':theta1, 'Theta2':theta2})
		print('Archivo Weights.Mat guardado existosamente! ')

	# Metodo para generar aleatorios (definido por epsilon)
	@classmethod
	def randInitWeights(cls, L_in, L_out):
		epsilon = math.sqrt(6)/math.sqrt(L_in + L_out)
		W = np.zeros((L_out, (1+L_in)))
		W = np.dot(np.random.rand(L_out, (1+L_in))*2, epsilon)-epsilon
		return W

	@classmethod
	def debugInitWeights(cls, fan_out, fan_in):
		length = fan_out * (fan_in + 1)
		# Inicializar W usando seno, esto asegura que siempre sean del mismo valor
		return np.reshape(np.sin(np.arange(1, length+1)), (fan_out, fan_in + 1), order='F') / 10	

	# Metodo para setear nuevos valores
	@classmethod
	def setNewValues(cls, inputL, hiddenL, outputL):
		self.inputL_size = inputL
		self.hiddenL_size = hiddenL
		self.outputL_size = outputL

	@classmethod
	def unroll_thetas(cls, nn_params, input_layer_size, hidden_layer_size, num_labels):
		theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, (input_layer_size + 1)), order='F')
		theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],
                        (num_labels, (hidden_layer_size + 1)), order='F')

		return theta1, theta2

