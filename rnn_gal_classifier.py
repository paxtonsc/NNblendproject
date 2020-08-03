import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import classify_num_galaxies as c

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

import math
import sys
import os

# define global vars
GALS = 5
BATCH_SIZE = 64
INPUT_SIZE = 45
PATH = os.path.join('models', 'rnn_model.pt')


class GalDataset(Dataset):

	def __init__(self, train=True):
		"""
		__init__ takes in boolean of train or test 
		and creates a data set consisting of either
		80 percent of data (train) or 20 percent of
		data (test)
		"""

		x_file = os.path.join('params', 'gal_sim_params_with_permutations.txt')
		y_file = os.path.join('params', 'sep_num_found_with_permutations.txt')
	
		x = np.loadtxt(x_file)
		y = np.loadtxt(y_file)

		x = torch.from_numpy(x).float()
		y = (torch.from_numpy(y) - 1).long()

		x = c.normalize(x)
		#self.y = F.one_hot(y.to(torch.int64), GALS).float()
		n_samples = len(x)
		if train:
			self.x = x[0:int(0.8*n_samples),:]
			self.y = y[0:int(0.8*n_samples)]
			self.n_samples = int(0.8*n_samples)
		else:
			self.x = x[int(0.8*n_samples):,:]
			self.y = y[int(0.8*n_samples):]
			self.n_samples = int(0.2*n_samples)


	def __getitem__(self, index):
		"""
		returns item in dataset at index
		"""
		return self.x[index], self.y[index]

	def __len__(self):
		"""
		returns length of dataset
		"""
		return self.n_samples


class RNNClassifier(nn.Module):
    def __init__(self):
    	super(RNNClassifier, self).__init__()
    	"""
    	initiates classifier, takes in 9 galaxy parameters as input
    	has one hidden layer, and outputs dim 5. Output is contains probability of 
    	1 - 5 galaxies being predicted by source extractor.
    	"""
    	self.rnn = nn.LSTM(input_size=9, hidden_size=64, num_layers=1, batch_first=True)
    	self.fc = nn.Linear(64, 5)


    def forward(self, x):
    	"""
    	propogates forward on input x,
    	output is returned at final ieration
    	"""

    	output, (h_n, h_c) = self.rnn(x, None)

    	# use output of last layer as input for linear layer
    	fc_out = self.fc(output[:, -1, :])
    	return fc_out


"""
Each input should be the parameters of a single galaxy
between 1 and five sets on input parameters

"""
def NN_vs_source(y_source, y_NN, GALS=5, PARAM_PER_GAL=9):
	"""
	graphs the NN output vs the true source extractor output
	"""

	class_array = np.zeros((GALS+1, GALS+1))
	for i in range(len(y_source)):
		class_array[y_source[i]+1][y_NN[i]+1] += 1

	class_array = (class_array.T/class_array.sum(axis=1)).T

	plt.matshow(class_array, cmap='YlOrRd')
	plt.ylabel('# identified by SE')
	plt.xlabel('# predicted by RNN')
	plt.suptitle('RNN predictions of SE vs SE')
	for (i, j), z in np.ndenumerate(class_array):
		plt.text(j, i, '{:0.3f}'.format(z), ha='center', va='center')
	file_name = os.path.join('charts', 'nn_vs_se_rnn.png')
	plt.savefig(file_name)
	plt.show()

		


def train():
	"""
	creates dataloader objects and instance of RNN to train model.
	saves model at specifified path
	"""

	train_set = GalDataset()
	test_set = GalDataset(train=False)
	train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, num_workers=2)

	# create RNN model
	rnn = RNNClassifier()
	optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)
	loss_func = nn.CrossEntropyLoss()

	# set hyper parameters
	num_epochs = 2
	total_samples = len(train_set)
	n_iterations = math.ceil(total_samples/BATCH_SIZE)
	print(total_samples, n_iterations)

	test_x = Variable(test_set.x, volatile=True).type(torch.FloatTensor)
	test_y = test_set.y.numpy().squeeze()

	print(f'length of test set y {len(test_y)}')

	for epoch in range(num_epochs):
		for i, (x, y) in enumerate(train_loader):

			# reshape x input values to (batch, time_step, input_size)
			b_x = Variable(x.view(-1, 5, 9))
			b_y = Variable(y)

			output = rnn(b_x)

			loss = loss_func(output, b_y)
			optimizer.zero_grad()

			loss.backward()

			optimizer.step()

			if i % 100 == 0:
				
				test_output = rnn(test_x.view(-1, 5, 9))
				pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
				accuracy = sum(pred_y == test_y) / float(test_y.size)

				print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_iterations}, inputs {x.shape}')
				print(f'accuracy: {accuracy}')


	torch.save(rnn.state_dict(), PATH)
	NN_vs_source(test_y, pred_y)
	


def eval():
	"""
	loads trained model and evaluates it
	"""

	test_set = GalDataset(train=False)

	test_x = Variable(test_set.x, volatile=True).type(torch.FloatTensor)
	test_y = test_set.y.numpy().squeeze()

	rnn = RNNClassifier()
	rnn.load_state_dict(torch.load(PATH))

	rnn.eval()

	test_output = rnn(test_x.view(-1, 5, 9))
	pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
	accuracy = sum(pred_y == test_y) / float(test_y.size)

	print(f'overall accuracy: {accuracy}')
	NN_vs_source(test_y, pred_y)


train()
#eval()
