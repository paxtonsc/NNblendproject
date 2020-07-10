import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
import os
import pandas as pd
from tqdm import tqdm


NUM_PARAMS = 4
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# return training data X, y and valid/test data X, y
def load_data():
	# locations of x input data and y out put data
	x_file = os.path.join('output', 'gal_sim_params.txt')
	y_file = os.path.join('output', 'accurate.txt')
	
	x = np.loadtxt(x_file)
	x = np.reshape(x, (int(x.shape[0]/2), x.shape[1]*2))
	y = np.loadtxt(y_file)
	y = y.reshape(len(y), 1)

	# break data into training and dev set
	x_train = x[0:9000,[0,1,9,10]]
	x_valid = x[9000:10000,[0,1,9,10]]
	y_train = y[0:9000,:]
	y_valid = y[9000:10000,:]


	print('x shape: ', x_train.shape)


	# attempt to create a more even data for training purposes
	# trying to avoid a 70% accuracy rate simply by classifying everything as a success

	# attempt to have a more even number of successes and failures
	# Do not want 70 percent accuracy by labeling everything as 1s
	#for i in range(9000):
	#	if y_train[i] == 0:
	#		x_train = np.append(x_train, x_train[i].reshape(1, len(x_train[i])), 0)
	#		y_train = np.append(y_train, y_train[i])
	#
	#y_train = y_train.reshape(len(y_train),1)

	return (x_train, y_train, x_valid, y_valid)


# simple NN class conisting of one hidden layer of 36 nodes
class Neural_Network(nn.Module):
	def __init__(self):
		super(Neural_Network, self).__init__()


		self.inputSize = NUM_PARAMS
		self.outputSize = 1
		self.hiddenSize = 24

		# weights
		self.W1 = torch.randn(self.inputSize, self.hiddenSize)
		self.W2 = torch.randn(self.hiddenSize, self.outputSize)


	def forward(self, X):
		self.z = torch.matmul(X, self.W1)
		self.z2 = self.sigmoid(self.z)
		self.z3 = torch.matmul(self.z2, self.W2)
		o = self.sigmoid(self.z3)
		return o
	
	def sigmoid(self, s):
		return 1/(1 + torch.exp(-s))
	
	def sigmoidPrime(self, s):
		return s * (1 - s)
	
	def backward(self, X, y, o):
		self.o_error = y - o
		self.o_delta = self.o_error * self.sigmoidPrime(o)
		self.z2_error = torch.matmul(self.o_delta, torch.t(self.W2))
		self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)

		alpha = 1e-4
		self.W1 += alpha * torch.matmul(torch.t(X), self.z2_delta)
		self.W2 += alpha * torch.matmul(torch.t(self.z2), self.o_delta)

	def train(self, X, y):
		o = self.forward(X)
		self.backward(X, y, o)
	
	def predict(self, X):
		return self.forward(X).round()
	
	def accuracy(self, y_pred, y):
		return 1 - torch.sum(torch.abs(y - y_pred))/len(y)
	


# collect data and turn into PyTorch tensors
X, y, X_valid, y_valid = load_data()
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).float()
y_valid = torch.from_numpy(y_valid).float()
X_valid = torch.from_numpy(X_valid).float()

NN = Neural_Network()
print("shape of y:", y.shape)


# train 
for i in range(5000):
	if (i%100 == 0):
		print('#' + str(i) + ' Loss: ' + str(torch.mean((y - NN(X))**2).detach().item()))
		print("accurarcy on training data: ",  NN.accuracy(NN.predict(X), y))
	NN.train(X, y)


# print out quant measures
y_pred = NN.predict(X_valid)
print('total number of 1s predicted in training data', torch.sum(NN.forward(X).round()))
print('total number of 1s predicted', torch.sum(y_pred))
print('accuracy on validation data set: ', NN.accuracy(y_pred, y_valid))

print('read data', y_valid[0:15,:])
print('predicted: ', y_pred[0:15,:])
