import numpy as np
import matplotlib.pyplot as plt
import yaml

with open('config.yaml') as config_file:
	config = yaml.load(config_file, Loader=yaml.FullLoader)


time_arr = np.loadtxt('result/time_arr.txt')
data_ekf = np.loadtxt('result/data_ekf.txt')
data_circular = np.loadtxt('result/data_circular.txt')
error_ekf = np.loadtxt('result/error_ekf.txt')
error_circular = np.loadtxt('result/error_circular.txt' )
R1_gt = np.loadtxt('result/Gt_date.txt')


## visualization
plot_color = {
	'EKF': config['color']['grenadine'],
	'LG-EKF': config['color']['mustard'],
	'hybrid': config['color']['navy'],
	'circular': config['color']['spruce']
}

plt.figure(1)
plt.plot(R1_gt[1:,1], R1_gt[1:,2], linewidth = 1.6, label = 'data_groundtruth')
plt.plot(data_ekf[:,0], data_ekf[:,1], "--", color = plot_color['EKF'], linewidth=1.6, label = 'EKF')
# plt.plot(data_hybrid[0:idx,0], data_hybrid[0:idx,1],'--',  color = plot_color['hybrid'], linewidth=1.6, label = 'hybrid')
plt.plot(data_circular[:,0], data_circular[:,1],'--',  color = plot_color['circular'], linewidth=1.6, label = 'circular')
# plt.plot(data_lie[:,0], data_lie[:,1],'--',  color = plot_color['LG-EKF'], linewidth=1.6, label = 'LG-EKF')
# plt.plot(groundtruth[:,0], groundtruth[:,1], linewidth = 1.6, label = 'groundtruth')

# plot the landmarks
# for idx in landmark_ids:
# 	plt.scatter(landmarks_list[idx][0],landmarks_list[idx][1])
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.legend()
plt.savefig('result/trajectory.png')


#Error plots
#Orientation error
plt.figure(2)
plt.subplot(211)
# plt.plot(time_arr, error_hybrid[:,0], color = plot_color['hybrid'], linewidth=1.6, label = 'hybrid')
plt.plot(time_arr, error_ekf[:,0], color = plot_color['EKF'], linewidth=1.6, label = 'EKF')

plt.plot(time_arr, error_circular[:,0], color = plot_color['circular'], linewidth=1.6, label = 'circular')
# plt.plot(time_arr, error_lie[:,0], color = plot_color['LG-EKF'], linewidth=1.6, label = 'Lie-EKF')
plt.ylabel('orientation err')
plt.legend()
#Positionerror
plt.subplot(212)
plt.plot(time_arr, error_ekf[:,1], color = plot_color['EKF'], linewidth=1.6)
# plt.plot(time_arr, error_hybrid[:,1], color = plot_color['hybrid'], linewidth=1.6)
plt.plot(time_arr, error_circular[:,1], color = plot_color['circular'], linewidth=1.6)
# plt.plot(time_arr, error_lie[:,1], color = plot_color['LG-EKF'], linewidth=1.6)
plt.ylabel('position err')

# plt.ylim([0, 0.3])
plt.show()
plt.show()
# # plt.pause(10)