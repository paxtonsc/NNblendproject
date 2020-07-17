"""
Creates randomized parameters for GalSim modeling
nine features: gal_flux, gal_sigma, psf_flux, psf_sigma, gal_x, gal_y, noise, e1, e2
"""

import numpy as np

def make_input_file(file_name, N, num_possible_gals):
	"""
	generates a matrix of params. First dimension is the number of systems.
	Secound dimension is the number of gals times number of params.

	:param file_name: location to store np.array
	:param N: number of images to be drawn
	:param num_possible_gals: max possile gal in one image
	:return: np array
	"""
	
	params_per_gal = 9

	array = np.zeros((N, num_pos_gals * params_per_gal))
	
	for i in range(num_pos_gals):
		array[:,i*params_per_gal + 0] = np.random.uniform(1.e4, 1.e5, N)
		array[:,i*params_per_gal + 1] = np.random.uniform(0.1,0.5, N)
		array[:,i*params_per_gal + 2] = np.random.uniform(0.9,1.1,N)
		array[:,i*params_per_gal + 3] = np.random.uniform(0.4,0.6,N)
		array[:,i*params_per_gal + 4] = np.random.uniform(-2., 2., N)
		array[:,i*params_per_gal + 5] = np.random.uniform(-2., 2., N)
		array[:,i*params_per_gal + 6] = np.random.uniform(25, 35, N)
		array[:,i*params_per_gal + 7] = np.clip(np.random.normal(0, 0.2, N), -.99, .99)
		array[:,i*params_per_gal + 8] = np.clip(np.random.normal(0, 0.2, N), -.99, .99)

		assert(np.max(array[:,i* params_per_gal+7]) < 1 and np.min(array[:,i* params_per_gal+7]) > -1)
		assert(np.max(array[:,i* params_per_gal+8]) < 1 and np.min(array[:,i* params_per_gal+8]) > -1)
	
	header = "ncols		%s\n" % array.shape[1]
	header += "nrows	%s\n" % array.shape[0]
	header += "gal_flux, gal_sigma, psf_flux, psf_sigma, gal_x, gal_y, noise, e1, e2"
	
	np.savetxt(file_name, array, header=header, fmt="%1.2f")

	return array


N = 10000
num_pos_gals = 5
name = 'test_input.asc'

make_input_file(name, N, num_pos_gals)

