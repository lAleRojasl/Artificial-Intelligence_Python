#! /usr/bin/env python

from Tkinter import *
import tkMessageBox
from MNISTLoader import MNIST
from displayData import displayData
from calcWeights import calcWeights
from nnFuncionCosto import resetIterations,getFinalError,nnFuncionGradiente, nnFuncionCosto as funCost
import csv
from sigmoidGradient import sigmoidGradient
from verificarNNGradientes import verificarNNGradientes
from predecir import predecir
from scipy.optimize import fmin_cg
from calcWeights import calcWeights
import threading as THD
import numpy as np
import scipy
import scipy.io
import matplotlib.pylab as plt
import sys

try:
    import ttk
    py3 = 0
except ImportError:
    import tkinter.ttk as ttk
    py3 = 1

# =========== Inicializacion ==================
# ------------- Variables ---------------------
inputL_size  = 784  # Entradas: Imagenes 28x28
hiddenL_size = 25   # Capas escondidas: definidas por usuario (default 25)
outputL_size = 10   # Salidas: 10 posibles numeros (salida[0] es el 10)
trainPrcnt = 60
validPrcnt = 40
lmbdaGrad = 0.01
lmbdaCosto = 0.01
lmbdaTrain = 0.01
iterations_counter = dict(val=0)
TestAccuracy = 0
Samples = 0

# --------------- Clases ----------------------
mndata = MNIST('./dataset/')  # Carga Dataset
dData = displayData()         # Muestra un numero aleatorio del Dataset
calcW = calcWeights(inputL_size, hiddenL_size, outputL_size)    	

# --------------- Training y Test Datasets --------------
trainImg = None
trainLabel = None
testImg = None
testLabel = None
# ------------------ Matriz de pesos ------------------
mat = scipy.io.loadmat('Weights.mat')
initial_theta1 = mat['Theta1']
initial_theta2 = mat['Theta2']	
initial_nn_params = np.concatenate((initial_theta1.T.ravel(), initial_theta2.T.ravel()))
# ----------------- Resultado de Entrenamiento ----------------
final_theta1 = None
final_theta2 = None


def set_Tk_var():
	global HiddenEntryVar
	HiddenEntryVar = StringVar()
	global TrainPrcntEntryVar
	TrainPrcntEntryVar = StringVar()
	TrainPrcntEntryVar.set("60")
	global CostoLmbdaEntryVar
	CostoLmbdaEntryVar = StringVar()
	CostoLmbdaEntryVar.set("0.01")
	global GSLambdaEntryVar
	GSLambdaEntryVar = StringVar()
	global SGEntryVar
	SGEntryVar = StringVar()
	global TrainLmbdaVar
	TrainLmbdaVar = StringVar()
	TrainLmbdaVar.set("0.01")
	global TrainSamplesVar
	TrainSamplesVar = StringVar()
	TrainSamplesVar.set("50")
	global TrainIterVar
	TrainIterVar = StringVar()
	TrainIterVar.set("50")

# ----------------------------------------------------------
# --------------- Acciones de Botones ----------------------

# -- Cambiar Hidden Layer -- 
def ChangeHidden(self):
	global initial_nn_params, hiddenL_size, mat, initial_theta1, initial_theta2
	#Verificamos que la entrada sea un entero positivo
	try:
		tempHidden = int(HiddenEntryVar.get())
		if(tempHidden < 0): raise ValueError
	except ValueError:
		InsertInLog('ERROR--> El valor ingresado debe ser un entero positivo\n')
		return
	hiddenL_size = tempHidden
        w.HiddenL_Label1.configure(text=HiddenEntryVar.get())
	HiddenEntryVar.set(u"")
	# Generar el nuevo archivo de pesos
	calcW = calcWeights(inputL_size, hiddenL_size, outputL_size) 
	InsertInLog('\nValor de Hidden Layers cambiado...\n'+
		      '...Generando nuevo archivo de pesos Weights.mat\n')
	calcW.newWeightsFile()
	InsertInLog('--> Archivo Weights.mat creado correctamente.\n')
	# Cargar nuevos datos 
	mat = scipy.io.loadmat('Weights.mat')
	initial_theta1 = mat['Theta1']
	initial_theta2 = mat['Theta2']
	initial_nn_params = np.concatenate((initial_theta1.T.ravel(), initial_theta2.T.ravel()))
	InsertInLog("Nueva matriz de pesos Weights.mat cargada.\n")
	InsertInLog("--> Dimension - Theta1: "+str(np.shape(initial_theta1))+"\n")
	InsertInLog("--> Dimension - Theta1: "+str(np.shape(initial_theta2))+"\n")

# -- Mostrar Imagen Random -- 
def ShowRandomImage(self):
	global trainImg, trainLabel
	if(trainImg != None and trainLabel != None):
		dData.showOneRandom(trainImg, trainLabel)
	else:
		InsertInLog("\nERROR--> No hay ningun DataSet cargado\n")
				
	
# -- Cargar Dataset --
def LoadDataset(self):
	global trainImg, testImg
	reloadDS = True
	if(trainImg != None and testImg != None):
		reloadDS = tkMessageBox.askyesno("Recargar", "Ya hay un dataset cargado, desea recargarlo?")

	if(reloadDS):
		InsertInLog('\nCargando Dataset...\n')
		t = THD.Thread(target=LoadTraining)
		t.start()
		t2 = THD.Thread(target=LoadTesting)
		t2.start()	

def LoadTraining():
	global trainImg, trainLabel
	trainImg, trainLabel = mndata.load_training()
	InsertInLog('...Training Dataset cargado correctamente.\n'
			    +'--> Dimensiones: '+str(np.shape(trainImg))+'\n' )

def LoadTesting():
	global testImg, testLabel
	testImg, testLabel = mndata.load_testing()
	testImg = np.divide(testImg, 255.)
	InsertInLog('...Testing Dataset cargado correctamente.\n'
			    +'--> Dimensiones: '+str(np.shape(testImg))+'\n' )
# -- Insertar informacion al TextArea --
def InsertInLog(text):
        w.LogBox.configure(state=NORMAL)
    	w.LogBox.insert(INSERT, text)
        w.LogBox.configure(state=DISABLED)

def ReplaceInLog(text):
        w.LogBox.configure(state=NORMAL)
	w.LogBox.delete("end-1c linestart", "end")
    	w.LogBox.insert(INSERT, text)
        w.LogBox.configure(state=DISABLED)	


# -- Cambiar % de distribucion de Training y Validation Sets -----
def SetNewDistribution(self):
	global trainPrcnt, validPrcnt
	#Verificamos que la entrada sea un entero positivo entre 1 y 100
	try:
		tempPrcnt = int(TrainPrcntEntryVar.get())
		if(tempPrcnt < 1 or tempPrcnt > 99): raise ValueError
	except ValueError:
		TrainPrcntEntryVar.set(trainPrcnt)
		InsertInLog('ERROR--> El valor debe ser un entero entre 1 y 99\n')
		return
	trainPrcnt = tempPrcnt
	validPrcnt = 100-tempPrcnt
	w.ValidPrcntLabel.configure(text=str(validPrcnt))
	InsertInLog('\n--> Distribucion modificada a '+str(tempPrcnt)+'-'+str(100-tempPrcnt)+'\n')

def CalcCost(self):
	global trainImg, trainLabel
	if(trainImg != None and trainLabel != None):	
		InsertInLog("\nCalculando costo inicial... \n")
		t = THD.Thread(target=funcionCosto)
		t.start()
	else:
		InsertInLog("\nERROR--> No hay ningun DataSet cargado\n")

def funcionCosto():
	global trainImg, trainLabel, initial_nn_params
	global inputL_size, hiddenL_size, outputL_size, lmbdaCosto
	lmbdaCosto = float(CostoLmbdaEntryVar.get())
	J = funCost(initial_nn_params, inputL_size, hiddenL_size, outputL_size, trainImg, trainLabel, lmbdaCosto, False)
	w.CostoIniLabel.configure(text=str(J))
	InsertInLog("--> Valor J: "+str(J)+"\n")
	

def CalcSigmoidGrad(self):
	grad,numgrad = verificarNNGradientes()
	InsertInLog('\n\nVerificando NN Gradientes:\n')
	for i in range(len(grad)):
		InsertInLog(str(numgrad[i])+" <--> "+str(grad[i])+"\n")

	diff = np.linalg.norm(numgrad-grad) / np.linalg.norm(numgrad+grad)
	
	InsertInLog("\n\nDiferencia relativa (debe ser menor a 1e-9) : "+ str(diff)+"\n")

def TrainingStart(self):
	global trainImg, trainLabel
	if(trainImg != None and trainLabel != None):	
		InsertInLog("\nEmpezando entrenamiento con "+TrainIterVar.get()+" iteraciones... \n")
		t = THD.Thread(target=Train)
		t.start()
	else:
		InsertInLog("\nERROR--> No hay ningun DataSet cargado\n")

def Train():
	global trainImg, trainLabel, initial_nn_params, lmbdaTrain, iterations_counter, Samples
	global inputL_size, hiddenL_size, outputL_size, final_theta1,final_theta2, TestAccuracy

    	calcW = calcWeights(inputL_size, hiddenL_size, outputL_size) 

    	lmbdaTrain = float(TrainLmbdaVar.get())
    	max_iterations = int(TrainIterVar.get())
    	iterations_counter = dict(val=0)	
    	resetIterations()
	
	Samples = int(TrainSamplesVar.get())
	temptrainImg = np.asarray(trainImg[:Samples])
	temptrainImg = np.divide(temptrainImg, 255.)
	temptrainLabel = np.asarray(trainLabel[:Samples])
    	def show_progress(current_x):
        	iterations_counter['val'] += 1
        	progress = iterations_counter['val'] * 100 // max_iterations
        	ReplaceInLog('\r[{0}{1}] {2}% - Iteracion:{3}'.format(
            	'=' * (progress // 5),
            	' ' * ((104 - progress) // 5),
            	progress, iterations_counter['val']
        	))

    	# Fmincg!
    	nn_params = fmin_cg(
        	funCost,
		x0=initial_nn_params,
		args=(inputL_size, hiddenL_size, outputL_size, temptrainImg, temptrainLabel, lmbdaTrain, True),
		fprime=nnFuncionGradiente,
		maxiter=max_iterations,
		callback=show_progress
    	)
	InsertInLog("\nCompletado correctamente!")	
	InsertInLog("\n--> Error Final: "+str(getFinalError()))	
	
	final_theta1, final_theta2 = calcW.unroll_thetas(nn_params, inputL_size, hiddenL_size, outputL_size)

	# -- Calcular y mostrar precision de las predicciones -- 	
	pred = predecir(final_theta1, final_theta2, testImg)
	InsertInLog("\n\nPrediciendo con las "+str(np.shape(testImg)[0])+" muestras del Test-Set:\n")
	InsertInLog("\nMuestra de 20 Labels Reales: "+str(np.asarray(testLabel[0:20]) ))
	InsertInLog("\nMuestra de 20 Predicciones : "+str(pred[0:20]))
	# -- Guardamos la prediccion -- 
	TestAccuracy = np.mean(pred == testLabel) * 100
	InsertInLog('\n\nPrecision de las predicciones: '+ 
		    str(TestAccuracy)+"\n")

def SaveData(self):
	global Samples, hiddenL_size, lmbdaTrain, trainPrcnt, validPrcnt
	global iterations_counter, TestAccuracy
	finalError = getFinalError()
	with open('results.csv', 'a') as csvfile:
	   	spamwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
	    	spamwriter.writerow([Samples, hiddenL_size, lmbdaTrain, trainPrcnt, validPrcnt,iterations_counter['val'], finalError, TestAccuracy])
		InsertInLog("\n\nResultados guardados en results.csv!\n")

def ClearLog(self):
        w.LogBox.configure(state=NORMAL)
    	w.LogBox.delete('1.0',END)
        w.LogBox.configure(state=DISABLED)
	
    # --------------- Acciones de Botones ----------------------
    # ----------------------------------------------------------

def init(top, gui, *args, **kwargs):
	global w, top_level, root, hiddenL_size
	w = gui
	top_level = top
	root = top
	InsertInLog("Matriz de pesos Weights.mat precargada.\n")
	InsertInLog("--> Dimension - Theta1: "+str(np.shape(initial_theta1))+"\n")
	InsertInLog("--> Dimension - Theta1: "+str(np.shape(initial_theta2))+"\n")
	hiddenL_size = int(np.shape(initial_theta1)[0])
        w.HiddenL_Label1.configure(text=str(hiddenL_size))

def destroy_window():
	# Function which closes the window.
	global top_level
	top_level.destroy()
	top_level = None

if __name__ == '__main__':
	import button
	button.vp_start_gui()


