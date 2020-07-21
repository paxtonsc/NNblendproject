import numpy as np
import sep
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Ellipse, Circle
import os


POS_NUM_GAL = 5

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
		e1 = real_params[POS_NUM_GAL*i + j][3]
		e2 = real_params[POS_NUM_GAL*i + j][4]
		r = 5*real_params[POS_NUM_GAL*i + j][1]

		h = np.hypot(e1, e2)
		beta = 0.5*np.arctan2(e2, e1)
		dx = h*np.cos(beta)
		dy = h*np.sin(beta)

		e = Ellipse(xy=(x, y), width=r, height=r, angle=beta*180/np.pi)
		e.set_facecolor('none')
		e.set_edgecolor('b')
		ax.add_artist(e)
	
	name = os.path.join('figures', 'figures2', 'figure%d' % i)
	plt.show()
	plt.savefig(name)
	plt.close()

def run_source_extractor(N, draw_figures=False):
	gal_params_file = os.path.join('params', 'gal_sim_params.txt')
	real_params = np.loadtxt(gal_params_file)
	position_data = np.zeros((POS_NUM_GAL*N, 2))
	num_found_data = np.zeros(N)
	
	for i in range(N):
	
		fits_file_name = os.path.join('blends', 'single_blend%d.fits' % i)
		data = fits.getdata(fits_file_name)
		data = data.byteswap(inplace=True).newbyteorder()
	
		bkg = sep.Background(data)
		# subtract background noise
		data_sub = data - bkg
	
		objects = sep.extract(data_sub, 1.5, err=bkg.globalrms)
	
		if (draw_figures):
			draw_figure(data_sub, objects)
	
		for j in range(len(objects)):
			position_data[2*i+j][0] = objects['x'][j]
			position_data[2*i+j][1] = objects['y'][j]
		
		num_found_data[i] = len(objects)
	
	if (not draw_figures):
		pos_file_name = os.path.join('params', 'sep_positions.txt')
		num_file_name = os.path.join('params', 'sep_num_found.txt')
		np.savetxt(pos_file_name, np.asarray(position_data))
		np.savetxt(num_file_name, num_found_data)
		print('updated training data')
	
run_source_extractor(100000)
#run_source_extractor(100, True)


