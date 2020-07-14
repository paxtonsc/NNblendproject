import sys
import os
import math
import numpy as np
import logging
import time
import galsim

def import_params():
	"""
	TODO: package input_maker.py and import to avoid this save and load
	step.
	"""

	cat_file_name = os.path.join('input', 'test_input.asc')
	cat = galsim.Catalog(cat_file_name)
	return cat


def draw_gal_sims(num=10000, num_gals_per=5, num_params=9, prob=0.5, xsize=50, ysize=50):
	"""
	function uses GalSim to draw images based on randomied parameters

	:param num: number of images to draw
	:param num_gal_per: max number of galaxies per draw
	:param num_params: number of params that define a single galaxy
	:param prob: probability of each possible additional galaxy being addded
	:param xsize: horizontal size of image in pixels
	:param ysize: vertical size of image in pixels
	
	:return tuple of image list and numpy array of param data
	that used in creation of the images
	"""

	images = []
	param_data = np.zeros((num*num_gals_per, num_params))
	pixel_scale = 0.2
	cat = import_params()
	file_name = os.path.join('output', 'blends', 'single_blend')

	for k in range(num):
		
		# save params
		params = np.zeros((num_gals_per, num_params))

		# add first galaxy
		params[0][6] = gal_flux = cat.getFloat(k, 0)
		params[0][2] = gal_sigma = cat.getFloat(k, 1)
		params[0][7] = psf_flux = cat.getFloat(k, 2)
		params[0][8] = psf_sigma = cat.getFloat(k, 3)
		params[0][0] = gal_x = cat.getFloat(k, 4)
		params[0][1] = gal_y = cat.getFloat(k, 5)
		params[0][5] = noise = cat.getFloat(k, 6)
		params[0][3] = e1_1 = cat.getFloat(k, 7)
		params[0][4] = e2_1 = cat.getFloat(k, 8)

		gal = galsim.Gaussian(flux=gal_flux, sigma=gal_sigma)
		
		gal = gal.shear(e1 = e1_1, e2 = e2_1)
		psf = galsim.Gaussian(flux=psf_flux, sigma=psf_sigma)

		final = galsim.Convolve([gal, psf])
		final = final.shift(gal_x, gal_y)

		image = galsim.ImageF(xsize, ysize)
		final.drawImage(image, scale=pixel_scale)

		image.addNoise(galsim.GaussianNoise(sigma=noise))
		param_data[num_gals_per*k] = params[0]

		# add up to num_gals_per -1 more galaxies to image
		for i in range(1,num_gals_per):
			coin = np.random.uniform(0,1,1)

			# 30 percent probaility of adding certain gal in
			if coin < prob:
							
				params[i][6] = gal2_flux = cat.getFloat(k, num_params*i + 0)
				params[i][2] = gal2_sigma = cat.getFloat(k, num_params*i + 1)
				params[i][7] = psf2_flux = cat.getFloat(k, num_params*i + 2)
				params[i][8] = psf2_sigma = cat.getFloat(k, num_params*i + 3)
				params[i][0] = gal2_x = cat.getFloat(k, num_params*i + 4)
				params[i][1] = gal2_y = cat.getFloat(k, num_params*i + 5)
				params[i][5] = noise2 = cat.getFloat(k, num_params*i + 6)
				params[i][3] = e1_2 = cat.getFloat(k, num_params*i + 7)
				params[i][4] = e2_2 = cat.getFloat(k, num_params*i + 8)

				gal2 = galsim.Gaussian(flux=gal2_flux, sigma=gal2_sigma)
				gal2 = gal2.shear(e1 = e1_2, e2 = e2_2)
				psf2 = galsim.Gaussian(flux=psf2_flux, sigma=psf2_sigma)

				final2 = galsim.Convolve([gal2, psf2])
				final2 = final2.shift(gal2_x, gal2_y)
				
				image = final2.drawImage(image=image, scale=pixel_scale, add_to_image=True)
				image.addNoise(galsim.GaussianNoise(sigma=noise2))

				param_data[num_gals_per*k + i] = params[i]


		image.write(file_name + '%s.fits' % k)
		images.append(image)
	
	return (images, param_data)

def main(argv):
	"""
	main function in blended.py. writes to logger, calls draw_gal_sims() to
	produce images, param_data and saves both to a text file

	:params argv:
	"""

	logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
	logger = logging.getLogger("Blended test")

	if not os.path.isdir('output'):
		os.mkdir('output')
	multi_file_name = os.path.join('output', 'multi_blend.fits')
	params_file_name = os.path.join('output','gal_sim_params.txt')

	logger.info('Simple test for created blended, randomized galaxies using: ')
	logger.info('	-Galaxy is guassian with parameters (flux, sigma) taken from catalog')
	logger.info('	-PSF also Gaussian with parameters taken from catalog')
	logger.info('	-Noise also Gaussian, with standard deviation (or variance) taken from catalog')
	logger.info('	-Number of galaxies range from 0 to 4 also taken from catalog, uniform distribution')
	logger.info('	-location of galxies (x,y) also from catalog from random distribution')

	images, param_data = draw_gal_sims(num_gals_per=2, prob=1)

	# for large number of files, writing to cube becomes unwieldy
	# galsim.fits.writeCube(images, multi_file_name)

	logger.info('Images written to multi-extension fits file %r', multi_file_name)
	logger.info('Created %s images', len(images))

	np.savetxt(params_file_name, param_data)	
	logger.info('wrote x,y, and size to %r', params_file_name)

if __name__ == "__main__":
	main(sys.argv)


