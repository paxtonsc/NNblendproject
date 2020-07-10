import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
import os


NUM_PARAMS = 20

def load_data():
	x_file = os.path.join('output', 'gal_sim_params.txt')
	y_file = os.path.join('output', 'accurate.txt')
	
	x = np.loadtxt(x_file)
	x = np.reshape(x, (int(x.shape[0]/2), x.shape[1]*2))
	y = np.loadtxt(y_file)

	# break data into training and dev set
	x_train = x[0:900,:]
	x_valid = x[900:,:]
	y_train = y[0:900]
	y_valid = y[900:1000]

	return (x_train, y_train, x_valid, y_valid)


def predict(X):
	x1 = X[:,0]
	y1 = X[:,1]
	x2 = X[:,9]
	y2 = X[:,10]

	deltax = x2 - x1
	deltay = y2 -y1
	dist = np.power((deltax*deltax + deltay*deltay), 0.5)

	return 1*(dist > 7.5)

	

X, y, X_valid, y_valid = load_data()

pred = predict(X)
print(pred[0:10])
print(y[0:10])
acc = 1 - np.sum(np.absolute(y - pred))/len(pred)
print("accurarcy on training data: ",  acc)


print('total number of 1s predicted in training data', np.sum(pred))
