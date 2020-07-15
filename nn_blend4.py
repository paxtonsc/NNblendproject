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
INPUT_DIM = 9
HIDDEN_DIM = 36				# somewhat arbitrarily choosen
NUM_LAYERS = 2
OUTPUT_DIM = 2
NUM_TRAIN = 100				# somewhat arbitrarily choosen

# set up tensorboard
writer = SummaryWriter()

# LSTM class
class LSTM(nn.Module):

	def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1):
		
		super(LSTM, self).__init__()
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		# number of training examples to train on each iteration
		self.batch_size = batch_size
		# number of reccurent layers. Default = 1, more than 1 results in a 
		# 'stacked' if more than 1
		self.num_layers = num_layers

		# LSTM layer
		self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

		# define output layer?
		self.linear = nn.Linear(self.hidden_dim, output_dim)
	
	def init_hidden(self):
		return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
				torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))
	
	def forward(self, input):
		lstm_out, self.hidden = self.lstm(input.view(len(input), self.batch_size, -1))
	
		# only take output from final time step
		# but generate at every time step?
		y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
	

# return training data X, y and test data X, y
def load_data():
	x_file = os.path.join('output', 'gal_sim_params.txt')
	y_file = os.path.join('output', 'sep_positions.txt')
	
	x = np.loadtxt(x_file)
	x = np.reshape(x, (int(x.shape[0]/2), x.shape[1]*2))
	y = np.loadtxt(y_file)

	y = y.reshape(int(y.shape[0]/2), y.shape[1]*2)
	
	print(y.shape)

	# break data into training and dev set
	x_train = x[0:9000,:]
	x_valid = x[9000:10000,:]
	y_train = y[0:9000]
	y_valid = y[9000:10000]

	return (x_train, y_train, x_valid, y_valid)

def round_tensor(t, decimal_place=3):
		return round(t.item(), decimal_place)

def normalize(x):
	return x/ x.max(0, keepdim=True)[0]


def train():
	# initates net object, constructs loss function and optimizer
	# optimizer updates parameters based on computed gradients
	# We implement Adam algorithm

	criterion = nn.MSELoss(size_average=False)
	optimizer = optim.Adam(NN.parameters(), lr=5e-5)

	prev_loss = 0
	hist = np.zeros(num_epochs)
	# train 
	for i in range(num_epochs):
		model.zero_grad()

		# Initialise hidden states
		model.hidden = model.init_hidden()

		y_pred = model(X_train)
		loss = loss_fn(y_pred, y_train)
		if i %100 == 0:
			print('Epoch ', i, 'MSE: ', loss.item())
		hist[i] = loss.item()

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

# collect data and turn into PyTorch tensors
X, y, X_test, y_test = load_data()
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).float()
y_test = torch.from_numpy(y_test).float()
X_test = torch.from_numpy(X_test).float()

X = normalize(X)
X_test = normalize(X_test)

model = LSTM(INPUT_DIM, HIDDEN_DIM, batch_size=NUM_TRAIN, output_dim=OUTPUT_DIM, num_layers=NUM_LAYERS)
train()
