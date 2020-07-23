import numpy as np
from classify_num_galaxies import *

def test_load_data():
	x_train, y_train, x_test, y_test = load_data()

	# check shape of data
	assert x_train.shape == (int(0.9 *N), NUM_PARAMS), "train data wrong shape"
	assert x_test.shape == (int(0.1 *N), NUM_PARAMS), "test data wrong shape"
	assert y_train.shape == (int(0.9 *N), GALS), "output train data wrong shape"
	assert y_test.shape == (int(0.1 *N), GALS), "output test data wrong shape"

	# test normalize feature
	assert(torch.max(x_train) < 20 and torch.min(x_train) > -20)
