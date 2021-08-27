import numpy as np
import matplotlib.pyplot as plt
import yaml
from scipy.stats import norm


def plot_results(Dataset_num, Robot_num):

	with open('config.yaml') as config_file:
		config = yaml.load(config_file, Loader=yaml.FullLoader)


	pic_resolution = 300

	#Load Sim Results
	time_arr = np.loadtxt('result/Robot{0}_dataset{1}_datatime_arr.txt'.format(Robot_num, Dataset_num))
	data_ekf = np.loadtxt('result/Robot{0}_dataset{1}_data_ekf.txt'.format(Robot_num, Dataset_num))
	data_circular =  np.loadtxt('result/Robot{0}_dataset{1}_data_circular.txt'.format(Robot_num, Dataset_num))
	error_ekf = np.loadtxt('result/Robot{0}_dataset{1}_error_ekf.txt'.format(Robot_num, Dataset_num))
	error_circular = np.loadtxt('result/Robot{0}_dataset{1}_error_circular.txt'.format(Robot_num, Dataset_num))
	R1_gt = np.loadtxt('result/Robot{0}_dataset{1}_Gt_data.txt'.format(Robot_num, Dataset_num))


	## visualization parameters
	plot_color = {
		'EKF': config['color']['grenadine'],
		'LG-EKF': config['color']['mustard'],
		'hybrid': config['color']['navy'],
		'circular': config['color']['spruce']
	}

	fig1, ax1 = plt.subplots(1, 1, figsize=(10, 7))
	ax1.plot(R1_gt[1:,1], R1_gt[1:,2], linewidth = 1.6, label = 'data_groundtruth')
	ax1.plot(data_ekf[:,0], data_ekf[:,1], "--", color = plot_color['EKF'], linewidth=1.6, label = 'EKF')
	# axes1.plot(data_hybrid[0:idx,0], data_hybrid[0:idx,1],'--',  color = plot_color['hybrid'], linewidth=1.6, label = 'hybrid')
	ax1.plot(data_circular[:,0], data_circular[:,1],'--',  color = plot_color['circular'], linewidth=1.6, label = 'circular')
	# axes1.plot(data_lie[:,0], data_lie[:,1],'--',  color = plot_color['LG-EKF'], linewidth=1.6, label = 'LG-EKF')

	# plot the landmarks
	# for idx in landmark_ids:
	#   axes1.scatter(landmarks_list[idx][0],landmarks_list[idx][1])
	ax1.set_xlabel('x (m)')
	ax1.set_ylabel('y (m)')
	ax1.legend()
	fig1.savefig('result/plots/Robot{0}_dataset{1}_trajectory.png'.format(Robot_num, Dataset_num), dpi =pic_resolution)


	#Error plots

	#Orientation error
	fig2, ax2 = plt.subplots(2, 1, figsize=(10, 7))
	# plt.plot(time_arr, error_hybrid[:,0], color = plot_color['hybrid'], linewidth=1.6, label = 'hybrid')
	ax2[0].plot(time_arr, error_ekf[:,0], color = plot_color['EKF'], linewidth=1.6, label = 'EKF')

	ax2[0].plot(time_arr, error_circular[:,0], color = plot_color['circular'], linewidth=1.6, label = 'circular')
	# axes2.plot(time_arr, error_lie[:,0], color = plot_color['LG-EKF'], linewidth=1.6, label = 'Lie-EKF')
	ax2[0].set_ylabel('orientation err')
	ax2[0].legend()
	
	#Positionerror
	ax2[1].plot(time_arr, error_ekf[:,1], color = plot_color['EKF'], linewidth=1.6)
	# plt.plot(time_arr, error_hybrid[:,1], color = plot_color['hybrid'], linewidth=1.6)
	ax2[1].plot(time_arr, error_circular[:,1], color = plot_color['circular'], linewidth=1.6)
	# plt.plot(time_arr, error_lie[:,1], color = plot_color['LG-EKF'], linewidth=1.6)
	ax2[1].set_ylabel('position err')
	# plt.ylim([0, 0.3])
	# plt.show()
	# # plt.pause(10)
	fig2.savefig('result/plots/Robot{0}_dataset{1}_errors.png'.format(Robot_num, Dataset_num), dpi =pic_resolution)
	plt.close(fig1)
	plt.close(fig2)


def analyze_results(Dataset_num, Robot_num):
	with open('config.yaml') as config_file:
		config = yaml.load(config_file, Loader=yaml.FullLoader)

	## visualization parameters
	plot_color = {
		'EKF': config['color']['grenadine'],
		'LG-EKF': config['color']['mustard'],
		'hybrid': config['color']['navy'],
		'circular': config['color']['spruce']
	}


	pic_resolution = 300

	#Load Sim Results
	time_arr = np.loadtxt('result/Robot{0}_dataset{1}_datatime_arr.txt'.format(Robot_num, Dataset_num))
	data_ekf = np.loadtxt('result/Robot{0}_dataset{1}_data_ekf.txt'.format(Robot_num, Dataset_num))
	data_circular =  np.loadtxt('result/Robot{0}_dataset{1}_data_circular.txt'.format(Robot_num, Dataset_num))
	error_ekf = np.loadtxt('result/Robot{0}_dataset{1}_error_ekf.txt'.format(Robot_num, Dataset_num))
	error_circular = np.loadtxt('result/Robot{0}_dataset{1}_error_circular.txt'.format(Robot_num, Dataset_num))
	R1_gt = np.loadtxt('result/Robot{0}_dataset{1}_Gt_data.txt'.format(Robot_num, Dataset_num))


	bin_num = 30

	fig1, ax1 = plt.subplots(2,1, figsize=(10,  7))
	fig1.suptitle('Error Distribution of Outputs', fontsize=16)
	domain = np.linspace(-3,3,100)

	ax1[0].hist(error_ekf[:,0], bins=bin_num, density = True, label = "ekf orientation", color = plot_color['EKF'])
	ax1[0].hist(error_circular[:,0], bins=bin_num, density = True, label = "circular orientation", color = plot_color['circular'])

	ax1[1].hist(error_ekf[:,1], bins=bin_num, density = True, label = "ekf dist", color = plot_color['EKF'])
	ax1[1].hist(error_circular[:,1], bins=bin_num, density = True, label = "circular dist", color = plot_color['circular'])

	plt.legend()
	plt.show()










	






