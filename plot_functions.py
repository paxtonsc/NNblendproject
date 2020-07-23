import torch
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_data_distribution(X):
	X = torch.flatten(X)
	plt.hist(X, bins=20)
	plt.xlabel('data value')
	plt.ylabel('# occurances')

	fig_name = os.path.join('charts', 'data_distribution.png')
	plt.savefig(fig_name)
	plt.show()

def plot_loss_acc(acc_test, acc_train, acc_1=None, acc_2=None, acc_3=None, acc_4=None, acc_5=None):
	"""
	plots accuracy after the nn is done training
	saves in 'charts' folder
	
	:param acc
	:param acc_train
	"""

	if acc_3 and acc_2:
		plt.plot(acc_1, label='test accuracy 1')
		plt.plot(acc_2, label='test accuracy 2')
		plt.plot(acc_3, label='test accuracy 3')
		plt.plot(acc_4, label='test accuracy 4')
		plt.plot(acc_5, label='test accuracy 5')
	plt.plot(acc_test,label='test accuracy')
	plt.plot(acc_train, label='train accuracy')
	plt.xlabel('epochs (in hundreds)')
	plt.legend()
	file_name = os.path.join('charts', 'nn_accuracy.png')
	plt.savefig(file_name)
	plt.show()


def plot_class_distribution(y_vals, x_vals, NN=False, GALS=5, PARAM_PER_GAL=9):
	""" 
	create matrix viz of galsim num generated galaxies
	to number recgnoized by source extractor or number that nn thinks 
	source extractor will recognize.

	:param y_vals: true number predicted by SE
	:param x_vals: true parameters from GalSim
	"""

	matrix = np.zeros((6,6))
	num_drawn = np.zeros(GALS)
	j = 0
	for x in x_vals:
		num = 0
		for i in range(5):
			if x[PARAM_PER_GAL*i+1] or x[PARAM_PER_GAL*i]:
				num += 1
			else:
				break
		num_drawn[num -1] += 1
		index = torch.where(y_vals[j] == 1)[0]
		matrix[num][index+1] += 1
		j+=1
		assert(index+1 >0)
	
	matrix = (matrix.T/matrix.sum(axis=1)).T
	
	#sums = torch.sum(y_vals, dim=0)
	#gals = ['one', 'two', 'three', 'four', 'five']
	#plt.bar(gals, num_drawn, alpha=0.6, label='gal sim galaxies drawn')
	#plt.bar(gals, sums, alpha=0.6, label='Source extractor galaxies found')
	#plt.xlabel('number of galaxies')
	#plt.ylabel('number of ocurances in training set')
	#plt.suptitle('Training Data Makeup')
	#if (NN):
	#	plt.suptitle('NN predictions')
	#plt.legend()
	plt.matshow(matrix, cmap='YlOrRd')
	plt.xlabel('# identified by SE')
	plt.ylabel('# drawn by GalSim')
	plt.suptitle('Training Data Makeup')
	if (NN):
		plt.suptitle('NN predictions')
	for (i, j), z in np.ndenumerate(matrix):
		plt.text(j, i, '{:0.3f}'.format(z), ha='center', va='center')
	
	path = os.path.join('charts', 'data_makeup.png')
	if NN:
		path = os.path.join('charts', 'data_makeup.NN.png')
	
	plt.savefig(path)
	plt.show()