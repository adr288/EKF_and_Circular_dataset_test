import numpy as np

def get_start_end_indeces(data, start_time, end_time):
	start_time_index = np.where(data[:,0] == start_time) # get the index of the starting time
	end_time_index = np.where(data[:,0] == end_time)   # get the index of the ending time

	
	# print(data[start_time_index,0])
	return start_time_index[0].item(), end_time_index[0].item()



def get_grond_truth(input_time, grond_truth_vector):

	# print(grond_truth_vector[:,0])
	index = np.where(grond_truth_vector[:,0] == input_time) # get the index of gt at time t
	# print(index)
	# print(grond_truth_vector[index,:])

#get the first ans las observations done in this time period
def get_list_of_observations(data, start_time, end_time):

	start_time_index = np.where(data[:,0] >= start_time) # get the index of the starting time
	end_time_index = np.where(data[:,0] <= end_time)
	indices = np.intersect1d(start_time_index,end_time_index)

	# print(indices)
	# print("Times are",data[indices,0])
	return data[indices] 





	
