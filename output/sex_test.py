import sewpy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from astropy.io import ascii
from astropy.io import fits
import os


N = 100
POS_NUM_GAL = 5

def run_sex(draw_figures=True):
	"""
	writes Source extractor output to a file named 'sew_out.dat'
	also records the x, y, and radius of every galaxy that source extractor identifies
	
	:params draw_figure: if true, appends the .fits images into a list to export annotated .pngs
	if false, program is able to run more quickly

	:return np.array: recording the profiles that Source Extractor identified per image and a 
	list of the images (which may be empty)
	"""

	f = open("sew_out.dat", "w+")
	
	sew = sewpy.SEW(params=["X_IMAGE", "Y_IMAGE", "FLUX_RADIUS(3)","ELLIPTICITY", "FLAGS"],
		config={"DETECT_MINAREA":10, "PHOT_FLUX":"0.3, 0.5, 0.8"})
	
	image_concat = []
	dimensions = np.zeros((N, 3*POS_NUM_GAL))
	
	for i in range(N):
		file_name = os.path.join("blends", "single_blend%d.fits" % i)
		out = sew(file_name)
		data = np.asarray(out["table"])

		x = []
		y = []
		r = []
		
		for j in range(data.shape[0]):
			dimensions[i][j*3] = data[j][0]
			dimensions[i][j*3+1]= data[j][1]
			dimensions[i][j*3+2] = data[j][2]

		if (draw_figures):
			image_concat.append(fits.getdata(file_name))

		ascii.write(out["table"], output=f)
	
	f.close()

	return (image_concat, dimensions)


def draw_plot(i, images, data, real_params):
	"""
	draws circles to represent the original galaxy profiles as rendered by galsim 
	and the recorded galaxy profiles as extracted by Source Extractor

	:params i: index
	:images list of .fits renders:
	:data (x,y,r) for source extractor identified image. Data per image is contained 
	a single row:
	:real_params original params used by GalSim:
	"""

	fig, ax = plt.subplots()
	ax.set_aspect('equal')
	ax.imshow(images[i], cmap='gray')

	
	for j in range(POS_NUM_GAL):
		x_sex = data[i][3*j + 0]-0.5
		y_sex = data[i][3*j + 1]-0.5
		r_sex = data[i][3*j + 2]

		ell = Circle((x_sex, y_sex), r_sex, facecolor='None', edgecolor='r', lw=2)
		ax.add_patch(ell)

	for j in range(POS_NUM_GAL):
		x_sim = real_params[POS_NUM_GAL*i+j][0] + 25
		y_sim = real_params[POS_NUM_GAL*i+j][1] + 25
		r_sim = real_params[POS_NUM_GAL*i+j][8]

		ell_sim = Circle((x_sim, y_sim), r_sim*2, facecolor='None', edgecolor='b', lw=2)
		ax.add_patch(ell_sim)

	name = os.path.join('figures', 'figure%d' % i)
	plt.savefig(name)
	plt.close()


def analyze(draw_images=True):
	# plot onto fits files
	#f = open("sew_out.dat", 'r')
	#data = f.read()

	images, data = run_sex(draw_images)
	real_params = np.loadtxt('gal_sim_params.txt')	
	accurate = np.ones(N)

	for i in range(N):
		# tracks wheather the same number of galaxies have been id by galsim and sex
		sex_accurate = 0

		# draw images if we want
		if draw_images:
			draw_plot(i, images, data, real_params)

	
	
analyze()
