import torch
import numpy as np
from torch.autograd import Variable
import os

x_file = os.path.join('output', 'gal_sim_params.txt')
y_file = os.path.join('output', 'accurate.txt')

x = np.loadtxt(x_file)
y = np.loadtxt(y_file)


