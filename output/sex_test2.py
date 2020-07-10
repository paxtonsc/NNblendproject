import sewpy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from astropy.io import ascii
from astropy.io import fits
import os



def run_sex():
	f = open("sew_out.dat", "w+")
	
	
	sew = sewpy.SEW(params=["X_IMAGE", "Y_IMAGE", "FLUX_RADIUS(3)","ELLIPTICITY", "FLAGS"],
		config={"DETECT_MINAREA":10, "PHOT_FLUX":"0.3, 0.5, 0.8"})
	
	image_concat = []
	dimensions = []
	
	for i in range(1000):
		file_name = os.path.join("blends", "single_blend%d.fits" % i)
		out = sew(file_name)
		data = np.asarray(out["table"])
	
		# save x, y data
		x1 = data[0][0]
		y1 = data[0][1]
		r1 = data[0][2]
		x2 = y2 = r2 = 0


		if (data.shape[0] == 2):
			x2 = data[1][0]
			y2 = data[1][1]
			r2 = data[1][2]
	
		dimensions.append((x1, y1, r1, x2, y2, r2))
	
	 	# image_concat.append(fits.getdata(file_name))
		# ascii.write(out["table"], output=f)
	
	
	#	f.close()
	return dimensions

def draw_plot(i, images, data, real_params):

	# creates graph
	fig, ax = plt.subplots()
	ax.set_aspect('equal')

	ax.imshow(images[i], cmap='gray')

	ell = Circle((data[i][0]-0.5, data[i][1]-0.5), data[i][2], facecolor='None', edgecolor='r', lw=2)
	ell_real = Circle((real_params[2*i][0] +25 , real_params[2*i][1] + 25), real_params[2*i][2], facecolor='None', edgecolor='b', lw=2)


	ax.add_patch(ell)
	ax.add_patch(ell_real)

	if (data[i][3]):
		ell2 = Circle((data[i][3]-0.5, data[i][4]-0.5), data[i][5], facecolor='None', edgecolor='r', lw=2, label='test')
		ax.add_patch(ell2)



	if (real_params[2*i+1][0]):
		ell_real2 = Circle((real_params[2*i+1][0] +25 , real_params[2*i+1][1] + 25), real_params[2*i+1][2], facecolor='None', edgecolor='b', lw=2)
		ax.add_patch(ell_real2)


	name = os.path.join('figures', 'figure%d' % i)
	plt.savefig(name)
	plt.close()


def analyze():
	# plot onto fits files
	#f = open("sew_out.dat", 'r')
	#data = f.read()

	data = run_sex()
	N = 1000
	real_params = np.loadtxt('gal_sim_params.txt')	
	accurate = np.ones(N)
	dist = []
	flux = []
	size = []

	dist_reg = []
	flux_reg = []
	size_reg = []


	for i in range(N):
		# tracks wheather the same number of galaxies have been id by galsim and sex
		sex_accurate = 0

		# draw images if we want
		# draw_plot(i, images, data, real_params)

		if (real_params[2*i+1][0]):
			sex_accurate +=1

		if (data[i][3]):
			sex_accurate -=1

		deltax = real_params[2*i][0] - real_params[2*i+1][0]
		deltay = real_params[2*i][1] - real_params[2*i+1][1]
		d = np.sqrt(deltax * deltax + deltay * deltay)
		dist_reg.append(d)

		flux_div = real_params[2*i][6]/real_params[2*i+1][6]
		if flux_div > 1:
			flux_div = 1.0/flux_div

		flux_reg.append(flux_div)
		size_reg.append(real_params[2*i+1][7])
		size_reg.append(real_params[2*i][7])

		if sex_accurate != 0:
			accurate[i] = 0
			dist.append(d)
			flux.append(flux_div)
			size.append(real_params[2*i][7])
			size.append(real_params[2*i+1][7])
	
	
	np.savetxt('accurate2.txt', accurate)
	plt.hist(dist_reg,bins=32, alpha=0.7, label='overall distribution')
	plt.hist(dist,bins=15, alpha=0.7, label='deblend failed')
	plt.xlabel('distance from centroids (pixels)')
	plt.ylabel('number of occurances')
	plt.legend()


	fig, ax = plt.subplots()

	plt.hist(flux_reg,bins=32, alpha=0.7, label='overall distribution')
	plt.hist(flux,bins=32, alpha=0.7, label='deblend failed')
	plt.xlabel('greater flux val/smaller flux val')
	plt.ylabel('number of occurances')
	plt.legend()

	flux_reg_av = np.average(flux_reg)
	flux_av = np.average(flux)

	plt.axvline(flux_reg_av, 0, 200, color='b')
	plt.axvline(flux_av, 0, 200, color='orange')

	fig, ax = plt.subplots()

	plt.hist(size_reg,bins=32, alpha=0.7, label='overall distribution')
	plt.hist(size,bins=32, alpha=0.7, label='deblend failed')
	plt.xlabel('size of SD in pixels')
	plt.ylabel('number of occurances')
	plt.legend()
	plt.show()
	
analyze()
