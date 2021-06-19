import yaml
import numpy as np
from scipy.io import loadmat
from util.dataset_util import get_start_end_indeces
from util.dataset_util import get_grond_truth
from util.dataset_util import get_list_of_observations

class Dataset:
	def __init__(self, path='Sampled_dataset/'):
		self.Robot  = 1
		self.start_time = 0
		self.durateion = 10 

		self.R_gt = 0
		self.R_obs = 0
		self.R_odom = 0
		self.landmark_ids = 0 
		self.landmarks_list = 0
		self.path = path



	def load_data(self, Robot):

		self.Robot = Robot
		with open('config.yaml') as config_file:
			config = yaml.load(config_file, Loader=yaml.FullLoader)
			sample_time = config['dt'] # 0.02

		#for number of existing robots
		self.R_gt = loadmat(self.path + 'Robot{}_Groundtruth.mat'.format(self.Robot))['Robot{}_Groundtruth'.format(self.Robot)]
		self.R_obs = loadmat(self.path + 'Robot{}_Measurement.mat'.format(self.Robot))['Robot{}_Measurement'.format(self.Robot)]
		self.R_odom = loadmat(self.path + 'Robot{}_Odometry.mat'.format(self.Robot))['Robot{}_Odometry'.format(self.Robot)]

		# self.landmarks_list = loadmat(self.path + 'Landmark_Groundtruth')
		# print("Langmarks are:", self.landmarks_list)


		self.landmark_ids =[72,27,54,70,36,18,25,9,81,16,90,61,45,7,63]

		self.landmarks_list = {63: [1.88032539, -5.57229508],
				  25: [1.77648406, -2.44386354],
				  45: [4.42330143, -4.98170313],
				  16: [-0.68768043, -5.11014717],
				  61: [-0.85117881, -2.49223307],
				  36: [4.42094946, -2.37103644],
				  18: [4.34924478, 0.25444762],
				  9:  [3.07964257, 0.24942861],
				  72: [0.46702834, 0.18511889],
				  70: [-1.00015496, 0.17453779],
				  81: [0.99953879, 2.72607308],
				  54: [-1.04151642, 2.80020985],
				  27: [0.34561556, 5.02433367],
				  7: [2.96594198, 5.09583446],
				  90: [4.30562926, 2.86663299]}


	def get_data(self, start_time=0, durateion = 400):
		self.start_time = start_time
		self.durateion = durateion

		#Round the time
		precision = 3  #based on the time step
		self.R_gt[:,0] = np.round(self.R_gt[:,0],precision)
		self.R_odom[:,0] = np.round(self.R_odom[:,0],precision)
		self.R_obs[:,0] = np.round(self.R_obs[:,0],precision)
		


		#Get the time stamps and find the start and ending indeces
		end_time = self.start_time + self.durateion
		start_idx, end_idx = get_start_end_indeces(self.R_gt, self.start_time, end_time)
		observations_input = get_list_of_observations(self.R_obs, self.start_time, end_time)
		
		odometry_input = self.R_odom[start_idx:end_idx]
		observation_input = self.R_obs[start_idx:end_idx]
		ground_truth = self.R_gt[start_idx:end_idx]

		number_of_observations = len(observations_input)
		total_sample_number = len(self.R_gt[start_idx:end_idx,0])
		data_size = total_sample_number + number_of_observations

		inital_state = ground_truth[0,1:4]
		inital_time = ground_truth[0,0]

		return inital_state, self.landmark_ids, self.landmarks_list, data_size, ground_truth, odometry_input, observations_input

	def simulation_data():
		pass
		
		
		

