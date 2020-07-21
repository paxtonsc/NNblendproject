import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

def accuracy(y_true, y_pred):
	"""
	:param y_true: accurate out put data
	:param y_pred: net work predicted output in form of probabilities of each classs
	:param verbose: if true prints out a graph of the data
	:param x: x-data is also required for verbose argument

	:return: 
	percent accuracy of times that the NN gives the correct class
	the highest probabilitiy
	"""

	max_el, max_idxs = torch.max(y_pred, dim=1) 
	predicted = F.one_hot(max_idxs, y_true.shape[1]).float()
	one = torch.ones(y_true.shape)
	return ((torch.logical_and(y_true, predicted)).sum().float()/len(y_true))

def classical_approach(X, dist):
	"""
	:param X: takes in full parameters but only uses to position

	returns preduction for y based off how many galaxies are dist
	apart from other galaxies
	"""
	x = X[:, [0,9,18,27,32]]
	y = X[:, [1,10,19,28,33]]


	gal_counter = torch.zeros(len(X))
	for n in range(len(X)):
		for i in range(1,5):
			if (x[n][i] != 0):
				gal_counter[n] += 1
		for i in range(5):
			for j in range(i+1, 5):
				deltax = x[n,i] - x[n,j]
				deltay = y[n,i] - y[n,j]
				d = np.sqrt(deltax*deltax + deltay*deltay)
				if x[n,i] == 0 or x[n,j] == 0:
					continue
				if d < dist:
					gal_counter[n] -= 1
					x[n,j] = 0
					
		
	gal_counter = F.one_hot(gal_counter.to(torch.int64), 5).float()
	return gal_counter

def classical_test(x, y):

	y_classical_accs = []
	dists = []
	for w in range(30):
		y_pred_c = classical_approach(x, float(w)/4)
		acc_c = accuracy(y, y_pred_c)
		y_classical_accs.append(acc_c)
		dists.append(float(w)/4)
	
	plt.plot(dists, y_classical_accs)
	plt.xlabel('cutoff distance between two galaxies (arcsecs)')
	plt.ylabel('accuracy')
	plt.title('accuracy of distance based algorithm')
	plt.savefig(os.path.join('charts', 'classical_acc.png'))