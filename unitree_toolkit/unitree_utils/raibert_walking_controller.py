import math
import numpy as np

from unitree_toolkit.unitree_utils.robot_config import a1_config
from unitree_toolkit.unitree_utils.quadruped_kinematics import quadruped_kinematics_solver

EPSILON = 1e-5

class Raibert_controller():
    def __init__(self, robot = 'a1', num_timestep_per_HL_action=100, target=None, speed_gain=0.0, des_body_ori=None,
                 control_frequency=100, leg_set_1=[0, 3], leg_set_2=[1, 2], leg_clearance=0.1, action_limit=None, **kwargs):
        if robot == 'a1':
            self.kinematics_solver = quadruped_kinematics_solver(**a1_config)
        
        self.control_frequency = control_frequency
        self.speed_gain = speed_gain 
        self.leg_clearance = leg_clearance
        self.num_timestep_per_HL_action = num_timestep_per_HL_action
        self.latent_action = None
        self.swing_set = leg_set_1
        self.stance_set = leg_set_2
        self.action_limit  = action_limit
        self.num_legs = self.kinematics_solver.num_legs
        self.n_dof = self.num_legs*self.kinematics_solver.num_motors_per_leg
        
        if target is None:
            self.target = np.array([0,0,0])
        else:
            self.target = target

        self.action_limit = np.zeros((self.n_dof, 2))
        if action_limit is None:
            self.action_limit[:, 0] = np.zeros(self.n_dof) + np.pi / 2.0
            self.action_limit[:, 1] = np.zeros(self.n_dof) - np.pi / 2.0

        if des_body_ori is None:
            self.des_body_ori = np.array([0,0,0]) # this is for des orientation at each timestep
        else:
            self.des_body_ori = des_body_ori
        self.final_des_body_ori = self.des_body_ori


    def set_init_state(self, init_state, action_limits=None):
        self.init_state = init_state
        if action_limits is not None:
            j_pos = np.array(self.init_state['j_pos'])
            self.action_limit[:, 0] = j_pos + np.array([action_limits] * self.num_legs).reshape(self.n_dof)
            self.action_limit[:, 1] = j_pos - np.array([action_limits] * self.num_legs).reshape(self.n_dof)
        self.set_control_params(init_state)


    def set_control_params(self, state):
        self.init_foot_pos_robot = self.kinematics_solver.forward_kinematics_robot(state['j_pos'])
        self.init_foot_pos = self.kinematics_solver.robot_frame_to_world_robot(state['base_ori_euler'], self.init_foot_pos_robot)
        self.standing_height = -(self.init_foot_pos[2] + self.init_foot_pos[5])/2.0
        self.des_body_ori = np.array([0, 0, state['base_ori_euler'][2]])
        self.final_des_body_ori = np.array([0, 0, state['base_ori_euler'][2]])
        self.init_r_yaw = self.get_init_r_yaw(self.init_foot_pos)
        self.swing_start_foot_pos_world = self.init_foot_pos


    def get_init_r_yaw(self,init_foot_pos):
        r_yaw = np.zeros((self.num_legs,2))
        for i in range(self.num_legs):
            r_yaw[i][0] = np.linalg.norm(init_foot_pos[3*i:3*i+2])
            r_yaw[i][1] = math.atan2(init_foot_pos[3*i+1], init_foot_pos[3*i])
        return r_yaw


    def plan_latent_action(self, state, target):
        self.latent_action = np.zeros(3)
        self.target = target
        current_speed = state['base_velocity'][0:2]

        speed_term = self.num_timestep_per_HL_action / (2 * self.control_frequency) * self.target[0:2]
        acceleration_term = self.speed_gain * (current_speed - self.target[0:2])
        orientation_speed_term = -self.num_timestep_per_HL_action / self.control_frequency * state['base_ori_euler'][2]

        des_footstep = (speed_term + acceleration_term)
        self.latent_action[0:2] = des_footstep
        self.latent_action[2] = orientation_speed_term
        return self.latent_action

    
    def switch_swing_stance(self):
        self.swing_set, self.stance_set = np.copy(self.stance_set), np.copy(self.swing_set)

    
    def update_latent_action(self, state, latent_action):
        self.switch_swing_stance()
        self.latent_action = latent_action

        swing_start_foot_pos_robot = self.kinematics_solver.forward_kinematics_robot(state['j_pos'])
        self.swing_start_foot_pos_world = self.kinematics_solver.robot_frame_to_world_robot(state['base_ori_euler'], swing_start_foot_pos_robot)

        self.last_com_ori = state['base_ori_euler']
        self.final_des_body_ori[2] = self.last_com_ori[2] + self.latent_action[-1]

        target_delta_xy = np.zeros(self.num_legs*3)
        for i in range(self.num_legs):
            if i in self.swing_set:
                angle = self.latent_action[-1] + self.init_r_yaw[i][1]
                target_delta_xy[3*i] = self.init_r_yaw[i][0] * math.cos(angle) + self.latent_action[0] - \
                                        swing_start_foot_pos_robot[3*i]
                target_delta_xy[3*i+1] = self.init_r_yaw[i][0] * math.sin(angle) + self.latent_action[1] - \
                                        swing_start_foot_pos_robot[3*i+1]
            else:
                angle = self.init_r_yaw[i][1]
                target_delta_xy[3*i] = self.init_r_yaw[i][0] * math.cos(angle) - self.latent_action[0] - \
                                        swing_start_foot_pos_robot[3*i]
                target_delta_xy[3*i+1] = self.init_r_yaw[i][0] * math.sin(angle) - self.latent_action[1] - \
                                        swing_start_foot_pos_robot[3*i+1]
            target_delta_xy[3*i+2] = -self.standing_height
        # transform target x y to the 0 yaw frame
        self.target_delta_xyz_world = self.kinematics_solver.robot_frame_to_world_robot(np.array(self.last_com_ori), target_delta_xy)


    
    def get_action(self, state, t):
        phase = float(t) / self.num_timestep_per_HL_action
        action = self._get_action(state, phase)
        action = np.clip(action, a_min=self.action_limit[:,1], a_max=self.action_limit[:,0])
        return action

    
    def _get_action(self, state, phase):
        self.des_foot_position_world = np.array([])
        self.des_body_ori[2] = (self.final_des_body_ori[2] - self.last_com_ori[2]) * phase + self.last_com_ori[2]
        # this seems to be designed only when walking on a flat ground
        des_foot_height_delta = (self.leg_clearance * math.sin(math.pi * phase + EPSILON))

        for i in range(self.num_legs):
            des_single_foot_pos = np.zeros(3)
            if i in self.swing_set:
                des_single_foot_pos[2] = 1.0*des_foot_height_delta - self.standing_height
            else:
                des_single_foot_pos[2] = 0.5*des_foot_height_delta - self.standing_height

            des_single_foot_pos[0] = self.target_delta_xyz_world[3*i] * phase + self.swing_start_foot_pos_world[3*i]
            des_single_foot_pos[1] = self.target_delta_xyz_world[3*i+1] * phase + self.swing_start_foot_pos_world[3*i+1]
            self.des_foot_position_world = np.append(self.des_foot_position_world ,des_single_foot_pos)


        self.des_foot_position_com = self.kinematics_solver.world_frame_to_robot_robot(self.des_body_ori, self.des_foot_position_world)
        des_leg_pose = self.kinematics_solver.inverse_kinematics_robot(self.des_foot_position_com)

        return des_leg_pose

    
    def reset(self, state):
        self.final_des_body_ori = np.array([0, 0, state['base_ori_euler'][2]])

 