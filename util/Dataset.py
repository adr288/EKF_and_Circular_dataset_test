import yaml
import numpy as np
from scipy.io import loadmat
from util.dataset_util import get_start_end_indeces
from util.dataset_util import get_grond_truth
from util.dataset_util import get_list_of_observations

class Dataset:
	def __init__(self, path='Sampled_dataset/'):
		self.Robot  = 1
		self.Dataset_num = 1
		self.start_time = 0
		self.durateion = 10 

		self.R_gt = 0
		self.R_obs = 0
		self.R_odom = 0
		self.landmark_ids = 0 
		self.landmarks_list = 0
		self.path = path
		self.sample_time = 0
		self.time_steps = 0



	def load_data(self, robot_num, dataset_num):
		'''
		Load the data from the .mat files smapled by the dataset matlab tools 
		'''

		self.Robot = robot_num
		self.Dataset_num = dataset_num
		with open('config.yaml') as config_file:
			config = yaml.load(config_file, Loader=yaml.FullLoader)
			sample_time = config['dt'] # 0.02


		print("Loading datset#{0} for Robot#{1}".format(self.Dataset_num, self.Robot))
		#Get the Sampling time and number of time steps captured and sampled in the dataset
		self.sample_time = loadmat(self.path + 'Dataset{}'.format(self.Dataset_num))['sample_time']
		self.time_steps = loadmat(self.path + 'Dataset{}'.format(self.Dataset_num))['timesteps']

		#Data collected from the robot
		self.R_gt = loadmat(self.path + 'Dataset{}'.format(self.Dataset_num))['Robot{}_Groundtruth'.format(self.Robot)]
		self.R_obs = loadmat(self.path + 'Dataset{}'.format(self.Dataset_num))['Robot{}_Measurement'.format(self.Robot)]
		self.R_odom = loadmat(self.path + 'Dataset{}'.format(self.Dataset_num))['Robot{}_Odometry'.format(self.Robot)]

		
		#Landmarks data
		self.landmarks_list = loadmat(self.path + 'Dataset{}'.format(self.Dataset_num))['Landmark_Groundtruth'][:,0:3]
		barcodes = loadmat(self.path + 'Dataset{}'.format(self.Dataset_num))['Barcodes']
		
		self.landmarks_list[:,0] = barcodes[5:,1] #List of the barcode numbers
		self.landmark_ids = barcodes[5:,1]   #i.e. [72,27,54,70,36,18,25,9,81,16,90,61,45,7,63]
		self.landmarks_list = dict(zip(self.landmarks_list[:,0].astype(int),self.landmarks_list[:,1:3])) #Turn the landmark data to dictionary {barcode: [landmark_x, landmark_y]}



	def get_data(self, start_time=0, duration = 400):
		'''
		returns the portion of the dataset with the specified start and duration times in seconds

		'''
		self.start_time = start_time
		if (duration == 'Full' and start_time == 0):
			print("Full duration seleted")
			self.duration = (self.sample_time * self.time_steps)[0,0] - 1
			print("Duration is {}".format(self.duration))
		
		else:
			self.duration = duration
			print("Duration is {}".format(self.duration))

		#Round the time
		precision = 3  #based on the time step's resolution
		self.R_gt[:,0] = np.round(self.R_gt[:,0],precision)
		self.R_odom[:,0] = np.round(self.R_odom[:,0],precision)
		self.R_obs[:,0] = np.round(self.R_obs[:,0],precision)
		
		#Get the time stamps and find the start and ending indeces
		end_time = self.start_time + self.duration
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
		
		
		

