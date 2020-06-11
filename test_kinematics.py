import math
import numpy as np
import matplotlib.pyplot as plt

from unitree_utils.robot_config import a1_config
from unitree_utils.quadruped_kinematics import quadruped_kinematics_solver

import unitree_env as e

kinematics_solver = quadruped_kinematics_solver(**a1_config)
env = e.UnitreeBasicEnv(render=True, robot ='a1',control_mode='position')
init_state = env.reset()

leg_index = 1
foot_pos_record = []
foot_pos_predict = []
joint_pos_record = []
joint_pos_predict = []
for _ in range(500):
    action = init_state['j_pos']
    action[2] = -2.0
    action[5] = -2.0
    state = env.step(action)
    foot_pos_record.append(state['foot_pos_robot'])
    joint_pos_record.append(state['j_pos'])
    
    # forward K
    foot_pos_robot = kinematics_solver.forward_kinematics_robot(state['j_pos'])
    foot_pos_world = kinematics_solver.robot_frame_to_world_robot(state['base_ori_euler'], foot_pos_robot)
    foot_pos_predict.append(foot_pos_world)

    # inverse K
    foot_pos_robot = kinematics_solver.world_frame_to_robot_robot(state['base_ori_euler'], state['foot_pos_robot'])
    joint_pos_robot = kinematics_solver.inverse_kinematics_robot(foot_pos_robot)
    joint_pos_predict.append(joint_pos_robot)


foot_pos_record = np.array(foot_pos_record)
foot_pos_predict = np.array(foot_pos_predict)
joint_pos_record = np.array(joint_pos_record)
joint_pos_predict = np.array(joint_pos_predict)
axis = range(500)
plt.rcParams['figure.figsize'] = (12, 10)
fig, ax = plt.subplots(2,3)
for i in range(3):
    ax[0,i].plot(axis, foot_pos_record[:,3*leg_index+i], label = 'Real')
    ax[0,i].plot(axis, foot_pos_predict[:,3*leg_index+i], label = 'Prediction')
    ax[1,i].plot(axis, joint_pos_record[:,3*leg_index+i], label = 'Real')
    ax[1,i].plot(axis, joint_pos_predict[:,3*leg_index+i], label = 'Prediction')
plt.legend()  
plt.show()
