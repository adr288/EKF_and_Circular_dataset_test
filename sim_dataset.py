# Localization trajectory
from util.dataset_simulation_function import run_selected_datasets_and_robots
from Plot_sim_results import plot_results


#Then loop through
for Dataset_num in range(1,2):
	for Robot_num in range(1,6):
		run_selected_datasets_and_robots(Dataset_num = Dataset_num, Robot_num = Robot_num, Duration = 'Full')
		plot_results(Dataset_num = Dataset_num, Robot_num = Robot_num)









