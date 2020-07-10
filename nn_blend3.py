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


# iniate psuedo random number generator
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# other Global vars
MODEL_PATH = 'model.pth'
NUM_PARAMS = 18


# set up tensorboard
writer = SummaryWriter()

# simple NN class conisting of two hidden layers
class Neural_Network(nn.Module):
	def __init__(self):
		super(Neural_Network, self).__init__()

		self.inputSize = NUM_PARAMS
		self.outputSize = 1
		self.hiddenSize1 = 36
		self.hiddenSize2 = 24

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
	x_file = os.path.join('output', 'gal_sim_params.txt')
	y_file = os.path.join('output', 'accurate.txt')
	
	x = np.loadtxt(x_file)
	x = np.reshape(x, (int(x.shape[0]/2), x.shape[1]*2))
	y = np.loadtxt(y_file)


	# break data into training and dev set
	x_train = x[0:9000,:]
	x_valid = x[9000:10000,:]
	y_train = y[0:9000]
	y_valid = y[9000:10000]

	return (x_train, y_train, x_valid, y_valid)


def accuracy(y_true, y_pred):
	predicted = y_pred.ge(.5).view(-1).float()
	return ((y_true == predicted).sum().float()/len(y_true))


def round_tensor(t, decimal_place=3):
		return round(t.item(), decimal_place)

def normalize(x):
	return x/ x.max(0, keepdim=True)[0]


def train():
	# collect data and turn into PyTorch tensors
	X, y, X_test, y_test = load_data()
	X = torch.from_numpy(X).float()
	y = torch.from_numpy(y).float()
	y_test = torch.from_numpy(y_test).float()
	X_test = torch.from_numpy(X_test).float()

	X = normalize(X)
	X_test = normalize(X_test)
	

	# initates net object, constructs loss function and optimizer
	# optimizer updates parameters based on computed gradients
	# We implement Adam algorithm
	NN = Neural_Network()
	criterion = nn.BCELoss()
	optimizer = optim.Adam(NN.parameters(), lr=5e-5)

	prev_loss = 0
	# train 
	for i in range(1000):
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

			print("#", i, " train loss ", round_tensor(train_loss), " accuracy: ", round_tensor(train_acc))
			print("Test accuracy: ", round_tensor(test_acc))
			
			# stop training if the loss has converged
			if torch.abs(prev_loss - train_loss) < 1e-3:
				break
			prev_loss = train_loss

		
		# use optimizer object to zero out grad
		# by default grads accumulate
		optimizer.zero_grad()

		# backward pass and update parameters
		train_loss.backward()
		optimizer.step()

	classes = ['did not recognize blend', 'recognize blend']
	y_pred = NN(X_test)
	y_pred = y_pred.ge(0.5).view(-1).cpu()
	y_test = y_test.cpu()


	print(classification_report(y_test, y_pred, target_names=classes))
	

	torch.save(NN, MODEL_PATH)

train()
