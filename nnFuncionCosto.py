import numpy as np
import scipy
import matplotlib.pylab as plt
from scipy.special import expit
from sigmoidGradient import sigmoidGradient
from calcWeights import calcWeights as calcW


def feedForward(Theta1, Theta2, trainImg):
	#Variables
	m = np.shape(trainImg)[0]
	# Feedforward

	A1 = np.array(np.hstack((np.ones((m, 1)), trainImg)))
  	Z2 = np.dot(A1, Theta1.conj().T)
	O1 = np.ones((np.shape(Z2)[0], 1))

	#EXPIT es la funcion SIGMOID de la libreria Scipy
	Sig = expit(Z2)
	ST1 = np.hstack( (O1, Sig) )
    	A2 = np.array( ST1 )
    	Z3 = np.dot(A2, Theta2.conj().T)
    	A3 = expit(Z3)

	return A1,A2,A3,Z2,Z3

def nnFuncionCosto(nn_params,inputL_size, hiddenL_size,
		   outputL_size, trainImg, trainLabel, lmbda):
	#Variables
	m = np.shape(trainImg)[0]

	Theta1, Theta2 = calcW.unroll_thetas(nn_params,inputL_size, hiddenL_size,outputL_size)

	A1, A2, A3, Z2, Z3 = feedForward(Theta1, Theta2, trainImg)
	
	# Trasformar etiquetas
	I = np.eye(outputL_size)
	Y = np.zeros((m, outputL_size))
    	for i in np.arange(0, m):
        	Y[i,:] = I[int(trainLabel[i]),:]
	
    	H = A3

	# Funcion de costo
   	J = np.dot(1./m, np.sum(np.sum((-Y*np.log(H)-(1.-Y)*np.log((1.-H))))))	

	# Regularizacion
    	P = np.dot(lmbda/(2.*m), np.sum(np.sum((Theta1[:,1:]**2.)))
					+np.sum(np.sum((Theta2[:,1:]**2.))) )

	# Funcion de costo con regularizacion
    	J = J+P
	#print("new cost: "+str(J))
	#print("Regularizacion (P):"+str(P))
	return J

def nnFuncionGradiente(nn_params, inputL_size, hiddenL_size, outputL_size, 
		       trainImg, trainLabel, lmbda):

	#Variables
	m = np.shape(trainImg)[0]

	Theta1, Theta2 = calcW.unroll_thetas(nn_params,inputL_size, hiddenL_size,outputL_size)

	A1, A2, A3, Z2, Z3 = feedForward(Theta1, Theta2, trainImg)


	#Variables a retornar
	Theta1_grad = np.zeros(np.shape(Theta1))
	Theta2_grad = np.zeros(np.shape(Theta2))

	# Trasformar etiquetas
	I = np.eye(outputL_size)
	Y = np.zeros((m, outputL_size))
    	for i in np.arange(0, m):
        	Y[i,:] = I[int(trainLabel[i]),:]
	

	# Error ultima capa
    	E3 = A3-Y
	# Error de capa 2
	E2 = np.dot(E3, Theta2)*sigmoidGradient(np.array(np.hstack((np.ones((np.shape(Z2)[0], 1)), Z2))))
	# Remover Bias
	E2 = E2[:,1:]
	
	# Acumular gradiente
	G1 = np.dot(E2.conj().T, A1)
	G2 = np.dot(E3.conj().T, A2)
	
	# Obtener D's regularizados
	Theta1_grad = (G1/m)+np.dot(lmbda/m, np.array( np.hstack(( np.zeros(( np.shape(Theta1)[0], 1 )), Theta1[:,1:] )) ) )
	Theta2_grad = (G2/m)+np.dot(lmbda/m, np.array( np.hstack(( np.zeros(( np.shape(Theta2)[0], 1 )), Theta2[:,1:] )) ) )
	grad = np.concatenate((Theta1_grad.T.ravel(), Theta2_grad.T.ravel()))
	
    	return grad
