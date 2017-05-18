import random
import numpy as np
import matplotlib.pyplot as plt

#Clase displayData
class displayData(object):
    	def __init__(self):
		# Plot con matplotlib
		plt.style.use('grayscale')
		self.dataMat = []

	#Funcion para mostrar la imagen de un numero aleatorio
	def showOneRandom(self, dataSetImg, dataSetLabel):
		fig, plot = plt.subplots(ncols=1)
		# Generamos un indice random para usar cualquier imagen del set
		imgIndex = random.randrange(0, len(dataSetLabel))

		# Titulo del plot
		fig.suptitle("Numero "+str(dataSetLabel[imgIndex])+" en Escala de Grises")

		# Agregamos la informacion de la imagen al plot
		self.dataMat = self.toMatrix(dataSetImg[imgIndex])

		# Agregamos los datos al plot
	        plot.imshow(self.dataMat, interpolation='none')
		# Finalmente lo mostramos
		plt.show()

	def showNumber(self, imgData, labelData):
		fig, plot = plt.subplots(ncols=1)

		# Titulo del plot
		fig.suptitle("Numero "+str(labelData)+" en Escala de Grises")

		# Agregamos la informacion de la imagen al plot
		self.dataMat = self.toMatrix(imgData)

		# Agregamos los datos al plot
	        plot.imshow(self.dataMat, interpolation='none')
		# Finalmente lo mostramos
		plt.show()


	# Funcion para convertir los datos (array) en una matriz.
	# La imangen viene como una lista de valores 0-255 Ej: 
	# ...0,0,0,0,0,25,60,100,250,120,25,0,0,0,0,0...
	# Necesitamos separarla en sublistas de 28 (pues la image es 28x28)
	@classmethod
	def toMatrix(cls, img, width=28):
	    finalMatrix = []
	    tempList = [img[0]]
	    for i in range(1,len(img)):
	    	if i % width == 0:
		    finalMatrix.append(tempList)
		    tempList = []
		    tempList.append(img[i])
		else:
		    tempList.append(img[i])

	    #agregamos la ultima fila
	    finalMatrix.append(tempList)
	    return finalMatrix

