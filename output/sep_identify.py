import numpy as np
import sep
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Ellipse, Circle
import os


def draw_figure(data_sub, objects):

	fig, ax = plt.subplots()
	m, s = np.mean(data_sub), np.std(data_sub)
	im = ax.imshow(data_sub, interpolation='nearest', cmap='gray',vmin=m-s, vmax=m+s,origin='lower')

	for j in range(len(objects)):
		e = Ellipse(xy=(objects['x'][j], objects['y'][j]),
					width=6*objects['a'][j], 
					height=6*objects['b'][j],
					angle=objects['theta'][j]*180. / np.pi)
		e.set_facecolor('none')
		e.set_edgecolor('red')
		ax.add_artist(e)
	
	for j in range(POS_NUM_GAL):
		x = 5*real_params[POS_NUM_GAL*i + j][0] + 25
		y = 5*real_params[POS_NUM_GAL*i + j][1] + 25 
		r = real_params[POS_NUM_GAL*i + j][8]

		e = Circle((x,y), r*8)
		e.set_facecolor('none')
		e.set_edgecolor('b')
		ax.add_artist(e)
	
	name = os.path.join('figures2', 'figure%d' % i)
	plt.savefig(name)
	plt.close()


N = 100
real_params = np.loadtxt('gal_sim_params.txt')
POS_NUM_GAL = 5
position_data = np.zeros((POS_NUM_GAL*N, 2))

for i in range(N):

	fits_file_name = os.path.join('blends', 'single_blend%d.fits' % i)
	data = fits.getdata(fits_file_name)
	data = data.byteswap(inplace=True).newbyteorder()

	bkg = sep.Background(data)
	# subtract background noise
	data_sub = data - bkg

	objects = sep.extract(data_sub, 1.5, err=bkg.globalrms)

	draw_figure(data_sub, objects)

	for j in range(len(objects)):
		position_data[2*i+j][0] = objects['x'][j]
		position_data[2*i+j][1] = objects['y'][j]

np.savetxt('sep_positions.txt', np.asarray(position_data))




