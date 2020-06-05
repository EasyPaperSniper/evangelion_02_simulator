
import numpy as np

class quadruped_kinematics_solver():
    def __init__(self, thigh_length, shank_length):
        # initial each leg start point position
        self.thigh_length = thigh_length
        self.shank_length = shank_length


    def robot_frame_to_world_robot(self,):
        pass

    def world_frame_to_world_robot(self,):
        pass

    def robot_frame_to_world_leg(self,):
        pass

    def world_frame_to_world_leg(self,):
        pass


    def forward_kinematics_robot(self, joint_pos):
        '''
        joint_pos: 12-dim 
        '''
        
        pass
    
    def forward_kinematics_leg(self):
        
        
        pass

    
    def inverse_kinematics_robot(self):
        pass

    
    def inverse_kinematics_leg(self):
        pass

