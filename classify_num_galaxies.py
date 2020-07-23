import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

from classical_classifier import classical_test
import plot_functions as p

# iniate psuedo random number generator
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# other Global vars
MODEL_PATH = 'model.pth'
GALS = 5
PARAM_PER_GAL = 9
NUM_PARAMS = GALS * PARAM_PER_GAL
N = 100000

# set up tensorboard for data viz
writer = SummaryWriter()

# simple NN class conisting of two hidden layers
class Neural_Network(nn.Module):
	def __init__(self):
		super(Neural_Network, self).__init__()

		self.inputSize = NUM_PARAMS
		self.outputSize = GALS
		self.hiddenSize1 = 64
		self.hiddenSize2 = 64

		self.fc1 = nn.Linear(self.inputSize, self.hiddenSize1)
		self.fc2 = nn.Linear(self.hiddenSize1, self.hiddenSize2)
		self.fc3 = nn.Linear(self.hiddenSize2, self.outputSize)
		
	def forward(self, X):
		X = F.relu(self.fc1(X))
		X = F.relu(self.fc2(X))
		X = torch.sigmoid(self.fc3(X))
		return X
	

# return training data X, y and test data X, y
def load_data():
	"""
	function loads data including gal sim paramters and
	sourcer extractor classifcations from set txt files.
	Turn the output data into one_hot format so that the neural
	net predicts probabilites of each number of identified galaxies.
	Breaks data into test and train set and returns both.
	"""

	x_file = os.path.join('params', 'gal_sim_params.txt')
	y_file = os.path.join('params', 'sep_num_found.txt')
	
	x = np.loadtxt(x_file)
	x = np.reshape(x, (int(x.shape[0]/GALS), x.shape[1]*GALS))
	y_data = np.loadtxt(y_file)

	x = torch.from_numpy(x).float()
	y_data = torch.from_numpy(y_data) - 1

	x = normalize(x)

	y = F.one_hot(y_data.to(torch.int64), GALS).float()
	
	# break data into training and dev set
	cut = int(0.9*N)
	x_train = x[0:cut,:]
	x_valid = x[cut:N,:]
	y_train = y[0:cut,:]
	y_valid = y[cut:N,:]


	return (x_train, y_train, x_valid, y_valid)


def accuracy(y_true, y_pred, verbose=False, x=None):
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
	if verbose:
		p.plot_class_distribution(predicted, x, True)
	return ((torch.logical_and(y_true, predicted)).sum().float()/len(y_true))


def round_tensor(t, decimal_place=3):
	"""
	:param t: tensor
	:param decimal_places: number of places to round to
	"""

	return round(t.item(), decimal_place)

def normalize(x):
	"""
	normalizes tensor x by dividing by std featurewise

	:param x: input tensor
	"""
	std = torch.std(x, dim=0)
	mean = torch.mean(x, dim=0)

	for i in range(len(x)):
		mask = (x[i] != 0)
		mean = mask.int() * mean

		x[i] = (x[i] - mean)/(std)

	p.plot_data_distribution(x)

	return x


def train():
	# collect data and turn into PyTorch tensors
	X, y, X_test, y_test = load_data()


	p.plot_class_distribution(y, X)
	# initates net object, constructs loss function and optimizer
	# optimizer updates parameters based on computed gradients
	# We implement Adam algorithm
	NN = Neural_Network()
	criterion = nn.MSELoss()
	optimizer = optim.Adam(NN.parameters(), lr=1e-3)

	prev_loss = 0
	acc_train_vec = []
	acc_vec = []
	
	# train 
	for i in range(2000):
		y_pred = NN(X)
		y_pred = torch.squeeze(y_pred)
		y_test_pred = NN(X_test)
		y_test_pred = torch.squeeze(y_test_pred)
		train_loss = criterion(y_pred, y)
		test_loss = criterion(y_test_pred, y_test)
		train_acc = accuracy(y, y_pred)
		test_acc = accuracy(y_test, y_test_pred)

		writer.add_scalars('Loss', {'Train loss': train_loss, 'Test loss': test_loss} , i)
		writer.add_scalars('Accuracy', {'Train accuracy': train_acc, 'Test accuracy': test_acc} , i)

		if (i%100 == 0):
			
			acc_vec.append(test_acc)
			acc_train_vec.append(train_acc)
		

			print("#", i, " train loss ", round_tensor(train_loss))
			print("#", i, " train acc ", round_tensor(train_acc), " test acc ", round_tensor(test_acc))
			
			# stop training if the loss has converged
			if torch.abs(prev_loss - train_loss) < 1e-4:
				break
			prev_loss = train_loss

		
		# use optimizer object to zero out grad
		# by default grads accumulate
		optimizer.zero_grad()

		# backward pass and update parameters
		train_loss.backward()
		optimizer.step()

	print(accuracy(y_test, y_test_pred, True, x=X_test))
	p.plot_loss_acc(acc_vec, acc_train_vec)
