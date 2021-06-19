# Localization trajectory

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from scipy.io import loadmat
from util.dataset_util import get_start_end_indeces
from util.dataset_util import get_grond_truth
from util.dataset_util import get_list_of_observations
import Agent
from util.Dataset import Dataset

### import parameters
with open('config.yaml') as config_file:
	config = yaml.load(config_file, Loader=yaml.FullLoader)

np.random.seed(1)
start_time = 100
durateion = 500
Robot_num = 1
#dataset_num = 9


#Get the dataset
dataset = Dataset()
dataset.load_data(Robot=Robot_num)
inital_state, landmark_ids, landmarks_list, data_size, R1_gt, R1_odom, observations = dataset.get_data(start_time=start_time, durateion = durateion)
do_observation = True



# Initiate the output data
time_arr = list()
data_ekf = np.zeros([data_size, 3])
data_hybrid = np.zeros([data_size, 3])
data_circular = np.zeros([data_size,3])
data_lie = np.zeros([data_size, 3])
error_ekf = np.zeros([data_size, 2])
error_circular = np.zeros([data_size, 2])

#initiate the robot
agent_1 = Agent.Agent(landmarks_list = landmarks_list, _theta=inital_state[2], _position=inital_state[0:2], _init_theta_given=True)

end_idx = len(R1_gt)
idx = 0
print("Start")
for i in range(0, end_idx, 1):
	t = R1_odom[i,0]
	
	if t % 10 == 0:
		print(t)
	# print(t)

	# print("odom time, gt time",R1_odom[i,0], R1_gt[i,0])
	agent_1.time_update(odom = R1_odom[i,1:3])
	time_arr.append(t)
	# print("time array len", len(time_arr))
	#Read the estimates
	# EKF
	[ekf_theta, ekf_x, ekf_y] = agent_1.EKF_estimate.read_estimation()
	data_ekf[idx,0] = ekf_x
	data_ekf[idx,1] = ekf_y
	data_ekf[idx,2] = ekf_theta
	error_ekf[idx] 	= agent_1.dataset_estimation_error(R1_gt[i,3], ekf_theta, R1_gt[i,1:3], [ekf_x, ekf_y])

	# hybrid
	# [hybrid_theta, hybrid_x, hybrid_y] = agent_1.hybrid_estimate.read_estimation()
	# data_hybrid[idx,0] = hybrid_x
	# data_hybrid[idx,1] = hybrid_y
	# circular
	[circular_theta, circular_x, circular_y] = agent_1.circular_estimate.read_estimation()
	data_circular[idx,0] = circular_x
	data_circular[idx,1] = circular_y
	data_circular[idx,2] = circular_theta
	error_circular[idx]  = agent_1.dataset_estimation_error(R1_gt[i,3], circular_theta, R1_gt[i,1:3], [circular_x, circular_y])# True orient, est_orient, true_pos, est_pos


	idx +=1
	# print("idx:",idx)

	observed_landmark_index = np.where(observations[:,0]== t)[0] #check if a landmark is seen at time t and how many
	if (len(observed_landmark_index)): #if a landmark index exists
		for j in observed_landmark_index: #loop through the landmarks seen at time t and do the observation updates(somtimes multiple landmarks are seen at a single time step)
			if observations[j,1] in landmarks_list and do_observation: #if the observed object is a landmark
				
				oderv_input = observations[j,1:4] #pass landmark ID, range, bearing
				
				# print("observation happend")
				agent_1.bd_observation_update(observ = oderv_input)
				time_arr.append(t) #same t as time update
				# print("time array len", len(time_arr))
				#ekf
				[ekf_theta, ekf_x, ekf_y] = agent_1.EKF_estimate.read_estimation()
				data_ekf[idx,0] = ekf_x
				data_ekf[idx,1] = ekf_y
				error_ekf[idx] 	= agent_1.dataset_estimation_error(R1_gt[i,3], ekf_theta, R1_gt[i,1:3], [ekf_x, ekf_y])
			
				# hybrid
				# [hybrid_theta, hybrid_x, hybrid_y] = agent_1.hybrid_estimate.read_estimation()
				# data_hybrid[idx,0] = hybrid_x
				# data_hybrid[idx,1] = hybrid_y

				# circular
				[circular_theta, circular_x, circular_y] = agent_1.circular_estimate.read_estimation()
				data_circular[idx,0] = circular_x
				data_circular[idx,1] = circular_y
				error_circular[idx]  = agent_1.dataset_estimation_error(R1_gt[i,3], circular_theta, R1_gt[i,1:3], [circular_x, circular_y])# True orient, est_orient, rtue_pos, est_pos

			
				idx +=1
				# print("idx:",idx)
		

np.savetxt('result/time_arr.txt', time_arr)
np.savetxt('result/data_ekf.txt', data_ekf[0:idx])
np.savetxt('result/data_circular.txt', data_circular[0:idx])
np.savetxt('result/error_ekf.txt', error_ekf[0:idx])
np.savetxt('result/error_circular.txt', error_circular[0:idx])
np.savetxt('result/Gt_date.txt', R1_gt[0:end_idx])



# ## visualization
# plot_color = {
# 	'EKF': config['color']['grenadine'],
# 	'LG-EKF': config['color']['mustard'],
# 	'hybrid': config['color']['navy'],
# 	'circular': config['color']['spruce']
# }

# plt.figure(1)

# plt.plot(R1_gt[1:end_idx,1], R1_gt[1:end_idx,2], linewidth = 1.6, label = 'data_groundtruth')
# plt.plot(data_ekf[0:idx,0], data_ekf[0:idx,1], "--", color = plot_color['EKF'], linewidth=1.6, label = 'EKF')
# # plt.plot(data_hybrid[0:idx,0], data_hybrid[0:idx,1],'--',  color = plot_color['hybrid'], linewidth=1.6, label = 'hybrid')
# plt.plot(data_circular[0:idx,0], data_circular[0:idx,1],'--',  color = plot_color['circular'], linewidth=1.6, label = 'circular')
# # plt.plot(data_lie[:,0], data_lie[:,1],'--',  color = plot_color['LG-EKF'], linewidth=1.6, label = 'LG-EKF')
# # plt.plot(groundtruth[:,0], groundtruth[:,1], linewidth = 1.6, label = 'groundtruth')

# # plot the landmarks
# # for idx in landmark_ids:
# # 	plt.scatter(landmarks_list[idx][0],landmarks_list[idx][1])

# plt.xlabel('x (m)')
# plt.ylabel('y (m)')
# plt.legend()
# plt.savefig('result/trajectory.png')

# plt.figure(2)
# plt.subplot(211)
# # plt.plot(time_arr, error_ekf[0:len(time_arr),0], color = plot_color['EKF'], linewidth=1.6, label = 'EKF')
# # plt.plot(time_arr, error_hybrid[:,0], color = plot_color['hybrid'], linewidth=1.6, label = 'hybrid')
# plt.plot(time_arr, error_circular[0:len(time_arr),0], color = plot_color['circular'], linewidth=1.6, label = 'circular')
# # plt.plot(time_arr, error_lie[:,0], color = plot_color['LG-EKF'], linewidth=1.6, label = 'Lie-EKF')
# plt.ylabel('orientation err')
# plt.legend()
# plt.subplot(212)
# # plt.plot(time_arr, error_ekf[0:len(time_arr),1], color = plot_color['EKF'], linewidth=1.6)
# # plt.plot(time_arr, error_hybrid[:,1], color = plot_color['hybrid'], linewidth=1.6)
# plt.plot(time_arr, error_circular[0:len(time_arr),1], color = plot_color['circular'], linewidth=1.6)
# # plt.plot(time_arr, error_lie[:,1], color = plot_color['LG-EKF'], linewidth=1.6)
# plt.ylabel('position err')

# # plt.ylim([0, 0.3])
# plt.show()
# plt.show()
# # # plt.pause(10)
