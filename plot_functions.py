import torch
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_data_distribution(X):
	#X = torch.flatten(X)
	X = X.reshape(X.shape[0]*5, int(X.shape[1]/5))

	fig, axs = plt.subplots(3, 3)
	for i in range(9):
		axs[int(i/3), i%3].hist(X[:,i], bins=20, alpha=0.4)

	axs[0, 0].set_title('x-pos from center')
	axs[0, 1].set_title('y-pos from center')
	axs[0, 2].set_title('gal std')
	axs[1, 0].set_title('e1')
	axs[1, 1].set_title('e2')
	axs[1, 2].set_title('noise')
	axs[2, 0].set_title('gal flux')
	axs[2, 1].set_title('psf flux')
	axs[2, 2].set_title('psf std')

	for ax in axs.flat:
		ax.set(xlabel = 'data value', ylabel = '# occurances')
		#ax.label_outer()

	fig.tight_layout(pad=0.3)


	fig_name = os.path.join('charts', 'data_distribution.png')
	plt.savefig(fig_name)

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


def NN_vs_source(y_source, y_NN, GALS=5, PARAM_PER_GAL=9):
	class_array = np.zeros((GALS+1, GALS+1))
	for i in range(len(y_source)):
		index1 = torch.where(y_source[i] == 1)[0]
		index2 = torch.where(y_NN[i] == 1)[0]
		class_array[index1+1][index2+1] += 1

	class_array = (class_array.T/class_array.sum(axis=1)).T

	plt.matshow(class_array, cmap='YlOrRd')
	plt.ylabel('# identified by SE')
	plt.xlabel('# predicted by NN')
	plt.suptitle('NN predictions of SE vs SE')
	for (i, j), z in np.ndenumerate(class_array):
		plt.text(j, i, '{:0.3f}'.format(z), ha='center', va='center')
	file_name = os.path.join('charts', 'nn_vs_se.png')
	plt.savefig(file_name)




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

