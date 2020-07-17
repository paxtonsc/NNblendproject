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
NUM_PARAMS = 45


# set up tensorboard
writer = SummaryWriter()

class EncoderRNN(nn.Module):
	def __init__(self, input_size, hidden_size):
		super(EncoderRNN, self).__init__()
		self.hidden_size = hidden_size

		self.gru = nn.GRU(input_size, hidden_size)
	
	def forward(self, input, hidden):
		output, hidden = self.gru(input, hidden)
		return output, hidden
	
	def initHidden(self):
		return torch.zeros(1, 1, self.hidden_size, device=device)

class DecodeRNN(nn.Module):
	def __init__(self, hidden_size, output_size):
		super(DecoderRNN, self).__init__()
		self.hidden_size = hidden_size

		self.gru = nn.GRU(output_size, input_size)
		self.out = nn.Linear(hidden_size, output_size)
		self.softmax = nn.LogSoftmax(dim=1)
	
	def forward(self, input, hidden):
		output = F.relu(input.view(1, 1, -1)
		output, hidden = self.gru(output, hidden)
		output = self.softmax(self.out(output[0]))
		return output, hidden
	
	def initHidden(self)
		return torch.zeros(1, 1, self.hidden_size, device=device)

# return training data X, y and test data X, y
def load_data():
	x_file = os.path.join('output', 'gal_sim_params.txt')
	y_file = os.path.join('output', 'sep_num_found.txt')
	
	x = np.loadtxt(x_file)
	x = np.reshape(x, (int(x.shape[0]/5), x.shape[1]*5))
	y_data = np.loadtxt(y_file)

	x = torch.from_numpy(x).float()
	y_data = torch.from_numpy(y_data) - 1


	x = normalize(x)
	y = F.one_hot(y_data.to(torch.int64), 5).float()
	
	# break data into training and dev set
	x_train = x[0:9000,:]
	x_valid = x[9000:10000,:]
	y_train = y[0:9000,:]
	y_valid = y[9000:10000,:]

	return (x_train, y_train, x_valid, y_valid)


def accuracy(x, y_true, y_pred, verbose=False):
	max_el, max_idxs = torch.max(y_pred, dim=1) 
	predicted = F.one_hot(max_idxs, y_true.shape[1]).float()
	one = torch.ones(y_true.shape)
	if verbose:
		print(predicted[0:10,:])
		plot_class_distribution(predicted, x)
	return ((torch.logical_and(y_true, predicted)).sum().float()/len(y_true))


def round_tensor(t, decimal_place=3):
		return round(t.item(), decimal_place)

def normalize(x):
	return x/ torch.sqrt(torch.std(x))

def plot_loss_acc(acc, loss_vec):
	plt.plot(acc,label='test accuracy')
	plt.plot(loss_vec, label='test loss')
	plt.xlabel('epochs (in hundreds)')
	plt.legend()
	plt.show()


def plot_class_distribution(y_vals, x_vals):
	matrix = np.zeros((6,6))
	num_drawn = np.zeros(5)
	j = 0
	for x in x_vals:
		num = 0
		for i in range(5):
			if x[9*i]:
				num += 1
			else:
				break
		num_drawn[num -1] += 1
		index = torch.where(y_vals[j] == 1)[0]
		matrix[num][index+1] += 1
		j+=1
	
	#matrix = (matrix.T/matrix.sum(axis=1)).T
	
	print(matrix)
	sums = torch.sum(y_vals, dim=0)
	gals = ['one', 'two', 'three', 'four', 'five']
	plt.bar(gals, num_drawn, alpha=0.6, label='gal sim galaxies drawn')
	plt.bar(gals, sums, alpha=0.6, label='Source extractor galaxies found')
	plt.xlabel('number of galaxies')
	plt.ylabel('number of ocurances in training set')
	plt.title('Training Data Makeup')
	plt.legend()
	plt.matshow(matrix, cmap='YlOrRd')
	plt.xlabel('number of galaxies identified by source extractor')
	plt.ylabel('number of galaxies drawn by GalSim')
	for (i, j), z in np.ndenumerate(matrix):
		plt.text(j, i, '{:0.3f}'.format(z), ha='center', va='center')
		
	plt.show()
	plt.close()


def train():
	# collect data and turn into PyTorch tensors
	X, y, X_test, y_test = load_data()

	plot_class_distribution(y, X)
	# initates net object, constructs loss function and optimizer
	# optimizer updates parameters based on computed gradients
	# We implement Adam algorithm
	NN = Neural_Network()
	criterion = nn.MSELoss()
	optimizer = optim.Adam(NN.parameters(), lr=1e-5)

	prev_loss = 0
	loss_vec = []
	accuracy_vec = []
	# train 
	for i in range(1000):
		y_pred = NN(X)
		y_pred = torch.squeeze(y_pred)
		y_test_pred = NN(X_test)
		y_test_pred = torch.squeeze(y_test_pred)
		train_loss = criterion(y_pred, y)
		test_loss = criterion(y_test_pred, y_test)
		train_acc = accuracy(X, y, y_pred)
		test_acc = accuracy(X_test, y_test, y_test_pred)
		
		
		writer.add_scalars('Loss', {'Train loss': train_loss, 'Test loss': test_loss} , i)
		writer.add_scalars('Accuracy', {'Train accuracy': train_acc, 'Test accuracy': test_acc} , i)

		if (i%100 == 0):
			
			accuracy_vec.append(test_acc)
			loss_vec.append(test_loss)

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

	print(accuracy(X_test, y_test, y_test_pred, True))
	print(y_test[0:10,:])
	plot_loss_acc(accuracy_vec, loss_vec)

train()
