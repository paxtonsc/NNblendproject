# goal of this progam is to write a input file for the mini project
# iniatlly we want 7 columns gal_flux, gal_sigma, psf_flux, psf_sigma, gal_x, gal_y, noise, e1, e2

import numpy as np


N = 10000
num_pos_gals = 5
params_per_gal = 9

array = np.zeros((N, num_pos_gals * params_per_gal))


for i in range(num_pos_gals):
	array[:,i*params_per_gal + 0] = np.random.uniform(8.e4, 5.e5, N)
	array[:,i*params_per_gal + 1] = np.random.uniform(1.0, 3.0, N)
	array[:,i*params_per_gal + 2] = np.ones(N)
	array[:,i*params_per_gal + 3] = np.random.uniform(0.5, 1.5, N)
	array[:,i*params_per_gal + 4] = np.random.uniform(-10., 10., N)
	array[:,i*params_per_gal + 5] = np.random.uniform(-10., 10., N)
	array[:,i*params_per_gal + 6] = np.random.uniform(25, 38, N)
	array[:,i*params_per_gal + 7] = np.random.normal(0, 0.2, N)
	array[:,i*params_per_gal + 8] = np.random.normal(0, 0.2, N)

header = "ncols		%s\n" % array.shape[1]
header += "nrows	%s\n" % array.shape[0]
header += "gal_flux, gal_sigma, psf_flux, psf_sigma, gal_x, gal_y, noise, e1, e2"

np.savetxt("test_input.asc", array, header=header, fmt="%1.2f")
