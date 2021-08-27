# Localization trajectory
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from scipy.io import loadmat

from util.dataset_util import get_start_end_indeces
from util.dataset_util import get_grond_truth
from util.dataset_util import get_list_of_observations
from util.Dataset import Dataset

import Agent

def run_dataset_test(agent, inital_state, landmark_ids, landmarks_list, data_size, Robot_gt, Robot_odom, observations, do_observation = True):
	# Initiate the output data
	time_arr = list()
	data_ekf = np.zeros([data_size, 3])
	data_hybrid = np.zeros([data_size, 3])
	data_circular = np.zeros([data_size,3])
	data_lie = np.zeros([data_size, 3])
	error_ekf = np.zeros([data_size, 2])
	error_circular = np.zeros([data_size, 2])


	end_idx = len(Robot_gt)
	idx = 0
	print("Start")
	for i in range(0, end_idx, 1):
		t = Robot_odom[i,0]
		
		if t % 10 == 0:
			print("Time {} sec".format(t))
		# print(t)

		# print("odom time, gt time",Robot_odom[i,0], Robot_gt[i,0])
		agent.time_update(odom = Robot_odom[i,1:3])
		time_arr.append(t)
		# print("time array len", len(time_arr))
		#Read the estimates
		# EKF
		[ekf_theta, ekf_x, ekf_y] = agent.EKF_estimate.read_estimation()
		data_ekf[idx,0] = ekf_x
		data_ekf[idx,1] = ekf_y
		data_ekf[idx,2] = ekf_theta
		error_ekf[idx] 	= agent.dataset_estimation_error(Robot_gt[i,3], ekf_theta, Robot_gt[i,1:3], [ekf_x, ekf_y])

		# hybrid
		# [hybrid_theta, hybrid_x, hybrid_y] = agent.hybrid_estimate.read_estimation()
		# data_hybrid[idx,0] = hybrid_x
		# data_hybrid[idx,1] = hybrid_y
		# circular
		[circular_theta, circular_x, circular_y] = agent.circular_estimate.read_estimation()
		data_circular[idx,0] = circular_x
		data_circular[idx,1] = circular_y
		data_circular[idx,2] = circular_theta
		error_circular[idx]  = agent.dataset_estimation_error(Robot_gt[i,3], circular_theta, Robot_gt[i,1:3], [circular_x, circular_y])# True orient, est_orient, true_pos, est_pos


		idx +=1
		# print("idx:",idx)

		observed_landmark_index = np.where(observations[:,0]== t)[0] #check if a landmark is seen at time t and how many
		if (len(observed_landmark_index)): #if a landmark index exists
			for j in observed_landmark_index: #loop through the landmarks seen at time t and do the observation updates(somtimes multiple landmarks are seen at a single time step)
				if observations[j,1] in landmarks_list and do_observation: #if the observed object is a landmark
					
					oderv_input = observations[j,1:4] #pass landmark ID, range, bearing
					
					# print("observation happend")
					agent.bd_observation_update(observ = oderv_input)
					time_arr.append(t) #same t as time update
					# print("time array len", len(time_arr))
					#ekf
					[ekf_theta, ekf_x, ekf_y] = agent.EKF_estimate.read_estimation()
					data_ekf[idx,0] = ekf_x
					data_ekf[idx,1] = ekf_y
					error_ekf[idx] 	= agent.dataset_estimation_error(Robot_gt[i,3], ekf_theta, Robot_gt[i,1:3], [ekf_x, ekf_y])
				
					# hybrid
					# [hybrid_theta, hybrid_x, hybrid_y] = agent.hybrid_estimate.read_estimation()
					# data_hybrid[idx,0] = hybrid_x
					# data_hybrid[idx,1] = hybrid_y

					# circular
					[circular_theta, circular_x, circular_y] = agent.circular_estimate.read_estimation()
					data_circular[idx,0] = circular_x
					data_circular[idx,1] = circular_y
					error_circular[idx]  = agent.dataset_estimation_error(Robot_gt[i,3], circular_theta, Robot_gt[i,1:3], [circular_x, circular_y])# True orient, est_orient, rtue_pos, est_pos

				
					idx +=1
					# print("idx:",idx)

	return time_arr, data_ekf, data_circular, error_ekf, error_circular, Robot_gt, idx, end_idx


def run_selected_datasets_and_robots(Dataset_num, Robot_num, Duration=100):
		### import parameters
	with open('config.yaml') as config_file:
		config = yaml.load(config_file, Loader=yaml.FullLoader)


	start_time = 0
	duration = Duration
	do_observation = True


	#Get the dataset
	dataset = Dataset()
	dataset.load_data(robot_num=Robot_num, dataset_num = Dataset_num)
	inital_state, landmark_ids, landmarks_list, data_size, R1_gt, R1_odom, observations = dataset.get_data(start_time=start_time, duration = duration)

	#initiate the robot
	agent_1 = Agent.Agent(landmarks_list = landmarks_list, _theta=inital_state[2], _position=inital_state[0:2], _init_theta_given=True)

	#run the dataset test
	time_arr, data_ekf, data_circular, error_ekf, error_circular, R1_gt, last_index, gt_last_idx = run_dataset_test(agent_1, inital_state, landmark_ids, landmarks_list, data_size, R1_gt, R1_odom, observations, do_observation = True)

	#Save the data
	np.savetxt('result/Robot{0}_dataset{1}_datatime_arr.txt'.format(Robot_num, Dataset_num), time_arr)
	np.savetxt('result/Robot{0}_dataset{1}_data_ekf.txt'.format(Robot_num, Dataset_num), data_ekf[0:last_index])
	np.savetxt('result/Robot{0}_dataset{1}_data_circular.txt'.format(Robot_num, Dataset_num), data_circular[0:last_index])
	np.savetxt('result/Robot{0}_dataset{1}_error_ekf.txt'.format(Robot_num, Dataset_num), error_ekf[0:last_index])
	np.savetxt('result/Robot{0}_dataset{1}_error_circular.txt'.format(Robot_num, Dataset_num), error_circular[0:last_index])
	np.savetxt('result/Robot{0}_dataset{1}_Gt_data.txt'.format(Robot_num, Dataset_num), R1_gt[0:gt_last_idx])



