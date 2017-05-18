from MNISTLoader import MNIST
from displayData import displayData
from calcWeights import calcWeights
from nnFuncionCosto import nnFuncionCosto as funCost
from sigmoidGradient import sigmoidGradient
from verificarNNGradientes import verificarNNGradientes
from trainer import trainer
from predecir import predecir
from predecir2 import predecir2
import numpy as np
import scipy
#import matcompat
import scipy.io
import matplotlib.pylab as plt
# =========== Inicializacion ==================
# ------------- Variables ---------------------
inputL_size  = 784  # Entradas: Imagenes 28x28
hiddenL_size = 25   # Capas escondidas: definidas pswor usuario (default 25)
outputL_size = 10   # Salidas: 10 posibles numeros (salida[0] es el 10)

# --------------- Clases ----------------------
mndata = MNIST('./dataset/')  # Carga Dataset
dData = displayData()         # Muestra un numero aleatorio del Dataset
calcW = calcWeights(inputL_size, hiddenL_size, outputL_size) 

# =========== Parte 1: visualizar =============
print("\n ======= Parte 1: Visualizar =======")
trainImg, trainLabel = mndata.load_training()
print('Training Dataset cargado correctamente!')
print('Dimensiones del training dataset: '+str(np.shape(trainImg)))
trainImg = np.asarray(trainImg[:2000])
trainImg = np.divide(trainImg, 255.)
trainLabel = np.asarray(trainLabel[:2000])

testImg, testLabel = mndata.load_testing()
print('Test Dataset cargado correctamente!')
print('Dimensiones del Test dataset: '+str(np.shape(testImg)))
testImg = np.asarray(testImg)
testImg = np.divide(testImg, 255.)
testLabel = np.asarray(testLabel)

#Mostrar dato random
#dData.showOneRandom(trainImg, trainLabel)
#dData.showNumber(trainImg[0], trainLabel[0])
raw_input("Presione ENTER para continuar...\n")

# ================ Parte 2: cargar parametros ================
print("\n ======= Parte 2: Cargar Parametros =======")
#print('Creando archivo nuevo archivo de Thetas aleatorios...\n')
#calcW.newWeightsFile() # Usar en caso de cambiar la cantidad de hidden layers

mat = scipy.io.loadmat('Weights.mat')
initial_theta1 = mat['Theta1']
initial_theta2 = mat['Theta2']	
initial_nn_params = np.concatenate((initial_theta1.T.ravel(), initial_theta2.T.ravel()))
print("Dimension - Theta1: "+str(np.shape(initial_theta1)))
print("Dimension - Theta2: "+str(np.shape(initial_theta2)))
raw_input("Presione ENTER para continuar...\n")


# ================ Parte 3: Calcular Costo (Feedfoward) ================
print("\n ======= Parte 3: Calcular Costo (Feedfoward) =======")
print("Feedforward usando Redes Neuronales...")

lmbda = 0
J = funCost(initial_nn_params, inputL_size, hiddenL_size, outputL_size, trainImg, trainLabel, lmbda)

print("Costo inicial (J): "+str(J))
raw_input("Presione ENTER para continuar...\n")

# ================ Parte 4: Regularizacion ================
print("\n ======= Parte 4: Regularizacion =======")
print("Verificando la funcion de costo (con regularizacion)... ")

#lmbda = 1
#J, grad = funCost(Theta1,Theta2, inputL_size, hiddenL_size, outputL_size, trainImg, trainLabel, lmbda)

#print("Costo de parametros (J): "+str(J))
raw_input("Presione ENTER para continuar...\n")

# ================ Parte 5: Sigmoid Gradient ================
print("\n ======= Parte 5: Sigmoid Gradient =======")
print("Evaluando SG...")
g = sigmoidGradient([1, -0.5, 0, 0.5, 1])
print("Sigmoid gradient para los valores [1, -0.5, 0, 0.5, 1]")
print(str(g))
raw_input("Presione ENTER para continuar...\n")

# ================ Parte 6: Implementando Backpropagation ================
print("\n ======= Parte 6: Implementando Backpropagation =======")
verificarNNGradientes()
raw_input("Presione ENTER para continuar...\n")

# ================ Parte 7: Training ================
print("\n ======= Parte 7: Training =======")
print("Entrenando con las primeras 2000 muestras del Training-Set...")
Theta1F, Theta2F = trainer(initial_nn_params, trainImg, trainLabel, inputL_size,hiddenL_size, outputL_size)

pred = predecir(Theta1F, Theta2F, trainImg)
print("\nRealizando predicciones usando TODO el Test-Set (m = 60000):\n")
print("Primeras 20 Predicciones: "+str(pred[1:20]))
print("Primeros 20 Labels Reales: "+str(trainLabel[1:20]))
print('\n\nPrecision de las predicciones: ', np.mean(pred == trainLabel) * 100)
print('Fin del programa...')

