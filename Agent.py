import numpy as np
import math
import yaml

from algorithms.EKF import EKF
from algorithms.LGEKF import LGEKF
from algorithms.hybrid import hybrid_loc_algo
from algorithms.circular import circular_loc_algo


with open('config.yaml') as config_file:
	config = yaml.load(config_file, Loader=yaml.FullLoader)



class Agent:
	def __init__(self, landmarks_list, _theta=0.0, _position=[0,0], _init_theta_given=True, _init_theta_cct=500.0):

		# parameter
		self.time = 0.0

		# state initialization
		self.theta = _theta 
		self.position = _position
		self.landmarks_list = landmarks_list 



		if _init_theta_given:
			initial_state = np.reshape(np.array([_theta, _position[0], _position[1]]),(3,1))
			initial_cov =np.array([[0.01,0,0], [0,0.01,0], [0,0,0.01]], dtype=float)

			# self.EKF_estimate = Estimators.GaussianSpatialState(_mean=initial_state, _cov=initial_cov)
			self.EKF_estimate = EKF(_mean=initial_state, _cov=initial_cov)
			# self.lie_estimate = LGEKF(_mean=initial_state, _cov=initial_cov) 
			# self.hybrid_estimate = hybrid_loc_algo(_phase=_theta, _concentration=1.0/0.01, _x=_position[0], _x_std=0.01, _y=_position[1], _y_std=0.01)
			self.circular_estimate = circular_loc_algo(_phase=_theta, _concentration=1.0/0.01, _x=_position[0],_y=_position[1])
			     

		else:                                    # unkown initial case (dynamic sim)
			theta_cov = 1.0 / _init_theta_cct 
			initial_state =  np.reshape(np.array([0, 0, 0]),(3,1))
			initial_cov =np.array([[theta_cov,0,0], [0,0.01,0], [0,0,0.01]], dtype=float)

			# self.EKF_estimate = Estimators.GaussianSpatialState(_mean=initial_state, _cov=initial_cov)
			self.EKF_estimate = EKF(_mean=initial_state, _cov=initial_cov)
			self.lie_estimate = LGEKF(initial_state, initial_cov)     
			self.hybrid_estimate = hybrid_loc_algo(_phase=0.0, _concentration=_init_theta_cct, _x=0.0, _x_std=0.01, _y=0.0, _y_std=0.01)
			self.circular_estimate = circular_loc_algo(_phase=0.0, _concentration=_init_theta_cct)




	def time_update(self, odom):

		dt = config['dt'] # s
		self.time += dt

		# angular velocity
		w = odom[1] 
		w_std = config['w_std'] 

		# translational velocity
		v = odom[0]
		v_std = config['v_std'] 
        
		real_v = v 
		real_w = w 

		# state update
		self.position[0] = self.position[0] + real_v * dt * math.cos(self.theta)
		self.position[1] = self.position[1] + real_v * dt * math.sin(self.theta)
		self.theta = (self.theta + real_w * dt) % (2*math.pi)

		# estimate update
		self.EKF_estimate.time_update(w, w_std, v, v_std, dt)
		# self.hybrid_estimate.time_update(w, w_std, v, v_std, dt)
		self.circular_estimate.time_update(w, w_std, v, v_std, dt)
		# self.lie_estimate.time_update(w, w_std, v, v_std, dt)


	def bd_observation_update(self, observ):

		
		Landmark_ID = int(observ[0])
		landmark = self.landmarks_list[Landmark_ID] # get the observed landmark's x,y position from its ID
		

		phi_cct = config['phi_cct'] 
		phi_std = 1.0 / math.sqrt(phi_cct) #0.01
		d_std = config['d_std']
		
		
		observ_bearing = observ[2] #% (2*math.pi)    #(math.atan2(dy, dx) - self.theta + np.random.vonmises(0, phi_cct)) % (2*math.pi)
		observ_distance = observ[1] # math.sqrt(dx**2 + dy**2) + np.random.normal(0, d_std)
		# print(observ_distance)

		# notice that some input is phi_std, and some is phi_cct
		self.EKF_estimate.bd_observation_update(landmark, observ_bearing, phi_std, observ_distance, d_std)
		# self.hybrid_estimate.bd_observation_update(landmark, observ_bearing, phi_cct, observ_distance, d_std)
		self.circular_estimate.bd_observation_update(landmark, observ_bearing, phi_cct, observ_distance, d_std)
		# self.lie_estimate.bd_observation_update(landmark, observ_bearing, phi_std, observ_distance, d_std)


	def dataset_estimation_error(self, true_theta, est_theta, true_pos, est_pos):
		or_error = 1 - math.cos(true_theta - est_theta)
		loc_error = math.sqrt((true_pos[0] - est_pos[0])**2 + (true_pos[1] - est_pos[1])**2)

		return [or_error, loc_error]
			

	def estimation_error(self, est_theta, est_x, est_y):

		or_error = 1 - math.cos(self.theta - est_theta)
		loc_error = math.sqrt((self.position[0] - est_x)**2 + (self.position[1] - est_y)**2)

		return [or_error, loc_error]


