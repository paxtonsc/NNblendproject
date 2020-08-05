import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import classify_num_galaxies as c

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

import math
import sys
import os

# define global vars
BATCH_SIZE = 30
EPOCH = 4
INPUT_SIZE = 9
TIME_STEP = 5
PATH = os.path.join('models', 'rnn_model.pt')
torch.manual_seed(1)



class GalDataset(Dataset):

	def __init__(self, train=True):
		"""
		__init__ takes in boolean of train or test 
		and creates a data set consisting of either
		80 percent of data (train) or 20 percent of
		data (test)
		"""

		x_file = os.path.join('params', 'gal_sim_params.txt')
		y_file = os.path.join('params', 'sep_num_found.txt')
		lens_file = os.path.join('params', 'lens.txt')
		
		x = np.loadtxt(x_file)
		y = np.loadtxt(y_file)
		lens = np.loadtxt(lens_file)

		print(x.shape)

		x = np.reshape(x, (int(x.shape[0]/5), x.shape[1]*5))
		x = torch.from_numpy(x).float()
		y = (torch.from_numpy(y) - 1).long()

		x = c.normalize(x)

		#self.y = F.one_hot(y.to(torch.int64), GALS).float()
		n_samples = len(x)
		print(n_samples)
		print(f'reshaped size of x', x.shape)

		
		if train:
			self.x = x[0:int(0.8*n_samples),:]
			self.y = y[0:int(0.8*n_samples)]
			self.n_samples = int(0.8*n_samples)
			self.lens = torch.from_numpy(lens[0:int(0.8*n_samples)])
		else:
			self.x = x[int(0.8*n_samples):,:]
			self.y = y[int(0.8*n_samples):]
			self.n_samples = int(0.2*n_samples)
			self.lens = torch.from_numpy(lens[int(0.8*n_samples):n_samples])


	def __getitem__(self, index):
		"""
		returns item in dataset at index
		"""
		return self.x[index], self.y[index], self.lens[index]

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
    	self.hidden_size = 64

    	self.rnn = nn.LSTM(INPUT_SIZE, 64, num_layers=1, batch_first=True)
    	self.out = nn.Linear(64, TIME_STEP)


    def forward(self, x, x_lengths):
    	"""
    	propogates forward on input x,
    	output is returned at final ieration

    	:params x: shape batch_size x time stipes x input dim
    	:params x_lengths: vector of size batch_size that contains number of stars
    	per image

    	"""
    	# when pack = True we have a varaible sized RNN
    	# otherwise everything is just padded
    	pack = False

    	batch_size, seq_len, _ = x.size()


    	#print('original size in forward', x.size())
    	if pack:
    		x = pack_padded_sequence(x, x_lengths, batch_first=True, enforce_sorted=False)

    	output, (h_n, h_c) = self.rnn(x, None)

    	if pack:
    		output, _ = pad_packed_sequence(output, batch_first=False, total_length=seq_len)
    		output = output.view(batch_size*seq_len, self.hidden_size)

    	"""
    	Idea is to look at the output value on the last non-padded input.
    	Previosuly we were always looking at the fith output regardless of
    	how many galaxies there were.

    	"""
    	if pack:
    		# uses the l-1 time step
    		adjusted_lengths = [batch_size*(l-1) + i for i, l in enumerate(x_lengths)]
    		lengthTensor = torch.tensor(adjusted_lengths, dtype=torch.int64)
    		output = output.index_select(0, lengthTensor)
    	elif not pack:
    		# uses that last output on the fith time step
    		output = output[:,-1,:]

    	#output = output.view(batch_size, self.hidden_size)

    	fc_out = self.out(output)

    	return fc_out





"""
Each input should be the parameters of a single galaxy
between 1 and five sets on input parameters

"""
def NN_vs_source(y_source, y_NN, GALS=5, PARAM_PER_GAL=9):
	"""
	graphs the NN output vs the true source extractor output
	"""

	class_array = np.zeros((TIME_STEP+1, TIME_STEP+1))
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

	# load training, test data
	train_set = GalDataset()
	test_set = GalDataset(train=False)
	train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE)
	test_x = Variable(test_set.x, volatile=True).type(torch.FloatTensor)
	test_y = test_set.y.numpy().squeeze()
	test_lens = test_set.lens.numpy().squeeze()

	# create RNN model
	rnn = RNNClassifier()
	print(rnn)
	optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001)
	loss_func = nn.CrossEntropyLoss()

	# set hyper parameters
	total_samples = len(train_set)
	n_iterations = math.ceil(total_samples/BATCH_SIZE)
	print(total_samples, n_iterations)

	for epoch in range(EPOCH):
		for i, (x, y, l) in enumerate(train_loader):

			# reshape x input values to (batch, time_step, input_size)
			b_x = Variable(x.view(-1, TIME_STEP, INPUT_SIZE))
			b_l = Variable(l)
			b_y = Variable(y)

			output = rnn(b_x, b_l)

			loss = loss_func(output, b_y)
			optimizer.zero_grad()

			loss.backward()

			optimizer.step()

			if i % 100 == 0:
				
				test_output = rnn(test_x.view(-1, TIME_STEP, INPUT_SIZE), test_lens)
				pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
				accuracy = sum(pred_y == test_y) / float(test_y.size)

				print(f'epoch {epoch+1}/{EPOCH}, step {i+1}/{n_iterations}, inputs {x.shape}')
				print(f'test accuracy: {accuracy}')
				print(f'loss {loss}')


	torch.save(rnn.state_dict(), PATH)
	NN_vs_source(test_y, pred_y)
	print(test_y[0:100])
	print(pred_y[0:100])
	


def eval():
	"""
	loads trained model and evaluates it
	"""

	test_set = GalDataset(train=False)

	test_x = Variable(test_set.x, volatile=True).type(torch.FloatTensor)
	test_y = test_set.y.numpy().squeeze()
	test_lens = test_set.lens.numpy().squeeze()


	rnn = RNNClassifier()
	rnn.load_state_dict(torch.load(PATH))

	rnn.eval()

	test_output = rnn(test_x.view(-1, TIME_STEP, INPUT_SIZE), test_lens)
	pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
	accuracy = sum(pred_y == test_y) / float(test_y.size)

	print(f'overall accuracy: {accuracy}')
	NN_vs_source(test_y, pred_y)


train()
#eval()
