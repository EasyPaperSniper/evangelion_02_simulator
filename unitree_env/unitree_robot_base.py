import os,  inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
parentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0,parentdir)

from unitree_pybullet.robot_bases import URDFBasedRobot

import numpy as np
import xmltodict as xdict
import pybullet
import time
import tempfile
import atexit

class UnitreeRobot(URDFBasedRobot):
    def __init__(self, robot_type = 'a1', initial_height = 0.40, self_collision=False,
        lateralFriction_robot=0.5, spinningFriction_robot=0.1, rollingFriction_robot = 0.1, linearDamping_robot = 0.1, angularDamping_robot = 0.1,
        restitution_robot=0.00, control_mode = 'position', initial_joint_positions=None, **kwargs):
        self._lateralFriction = lateralFriction_robot
        self._spinningFriction = spinningFriction_robot
        self._rollingFriction = rollingFriction_robot
        self._restitution = restitution_robot
        self._initialize_robot_subtype(robot_type, initial_height, self_collision)
        self._linearDamping = linearDamping_robot
        self._angularDamping = angularDamping_robot
        self.control_mode = control_mode
        self.initial_height = initial_height
        self._first_urdf_load = True

    

    def _initialize_robot_subtype(self, robot_type, initial_height, self_collision):
        if robot_type == 'a1':
            URDFBasedRobot.__init__(self, "a1.urdf", 'a1',
                        action_dim=3*4,
                        obs_dim=4, basePosition=[0, 0, initial_height], baseOrientation=[0, 0, 0, 1],
                        fixed_base=False, self_collision=self_collision)
            self._adapted_urdf_filepath = (os.path.dirname(os.path.dirname(currentdir)) + 
                                    '/evangelion_02_simulator/unitree_env/unitree_data/a1/a1_urdf/a1_eva_01.urdf')
            self._initial_joint_positions = [-0.05,0.60,-1.20,
                                             0.05, 0.60,-1.20,
                                             -0.05,0.65,-1.0,
                                             0.05, 0.65,-1.0]

        if robot_type == 'aliengo':
            URDFBasedRobot.__init__(self, "aliengo.urdf", "aliengo",
                        action_dim=3*4,
                        obs_dim=4, basePosition=[0, 0, initial_height], baseOrientation=[0, 0, 0, 1],
                        fixed_base=False, self_collision=self_collision)
            self._adapted_urdf_filepath = (os.path.dirname(os.path.dirname(currentdir)) + 
                                    '/evangelion_02_simulator/unitree_env/unitree_data/aliengo/aliengo_urdf/aliengo.urdf')
            self._initial_joint_positions =[-0.15,0.60,-1.20,
                                            0.15, 0.60,-1.20,
                                            -0.15,0.65,-1.0,
                                            0.15, 0.65,-1.0]

            
    def robot_specific_reset(self, bullet_client):
        """ This function is called for each reset of the environment.

        We do here specific resets for our robot. At the moment we do reload
        the whole environment instead of restoring a state due to issues
        with the pybullet simulator. This makes the reset a bit slower but
        avoids segmentation faults.

        Args:
            bullet_client: pointer to the pybullet client
        """
        if self._first_urdf_load:
            self._create_robot_from_urdf(bullet_client)
            self._first_urdf_load = False
        self.set_friction_and_restitution(bullet_client, lateralFriction=self._lateralFriction, restitution=self._restitution,
                                            linearDamping=self._linearDamping, angularDamping=self._angularDamping)
        self._reset_joint_positions()


    def _reset_joint_positions(self):
        for i in range(len(self.ordered_joints)):
            j = self.ordered_joints[i]
            j.reset_position(self._initial_joint_positions[i], 0.0)
            j.set_torque(0.0)

    
    def _create_robot_from_urdf(self, bullet_client):
        """Loads the current urdf file into the simulator.

        Args:
            bullet_client: pointer to the pybullet client
        """
        self.ordered_joints = []
        self.ordered_foot = []
        if self.self_collision:
            self.parts, self.jdict, self.ordered_joints, self.robot_body, self.ordered_foot = self.addToScene(bullet_client,
                bullet_client.loadURDF(self._adapted_urdf_filepath,
                basePosition=[0, 0, self.initial_height],
                baseOrientation=self.baseOrientation,
                useFixedBase=self.fixed_base,
                flags=pybullet.URDF_USE_SELF_COLLISION))
        else:
            self.parts, self.jdict, self.ordered_joints, self.robot_body, self.ordered_foot  = self.addToScene(bullet_client,
                bullet_client.loadURDF(self._adapted_urdf_filepath,
                basePosition=[0, 0, self.initial_height],
                baseOrientation=self.baseOrientation,
                useFixedBase=self.fixed_base))


    def set_control_mode(self, mode):
        """Sets the control mode

        Args:
            mode (string): 'velocity', 'position' or 'effort'

        Returns:
            bool: True if known control code
        """
        if mode in ['velocity', 'position', 'effort']:
            self.control_mode = mode
            return True
        # If we don't know it we just ignore it
        return False


    def apply_action(self, a):
        """Applies actions to the robot.

        Args:
            a (list): List of floats. Length must be equal to len(self.ordered_joints).
        """
        assert (np.isfinite(a).all())
        assert len(a) == len(self.ordered_joints)
        
        for n, j in enumerate(self.ordered_joints):
            action = a[n]
            if self.control_mode == 'velocity':
                # action = self._check_limits(j.joint_name, j, action)
                velocity = action
                # velocity = self._check_limits(j.joint_name, j, velocity)
                # velocity = np.clip(velocity, -1, +1)
                velocity = float(velocity)
                j.set_velocity(velocity)

            elif self.control_mode == 'position':
                position = action
                # position = self._check_limits(j.joint_name, j, position)
                j.set_position(float(np.clip(position, -np.pi, np.pi)))

            elif self.control_mode == 'effort':
                effort = action
                # effort = self._check_limits(j.joint_name, j, effort)
                j.set_torque(effort)
            else:
                print('not implemented yet')

    
    def calc_state(self):
        """Computes the state.

        Unlike the original gym environment, which returns only a single
        array, we return here a dict because this is much more intuitive later on.

        Returns:
        DaisyRobot    dict: The dict contains four different states. 'j_pos' are the
                joint positions. 'j_vel' are the current velocities of the
                joint angles. 'base_pos' is the 3D position of the base of Daisy
                in the world. 'base_ori_euler' is the orientation of the robot
                in euler angles.
        """
        joint_positions = [j.get_position() for j in self.ordered_joints]
        joint_velocities = [j.get_velocity() for j in self.ordered_joints]
        joint_effort = [j.get_torque() for j in self.ordered_joints]
        foot_pos = [f.current_position() for f in self.ordered_foot]
        base_position = self.robot_body.current_position()
        base_velocity = self.robot_body.speed()
        base_angular_velocity =self.robot_body.angular_velocity()
        self.body_xyz = base_position
        base_orientation_quat = self.robot_body.current_orientation()
        base_orientation_euler = self._p.getEulerFromQuaternion(base_orientation_quat)
        base_orientation_matrix= self._p.getMatrixFromQuaternion(base_orientation_quat)

        return {
        'base_pos_x' : base_position[0:1],
        'base_pos_y' : base_position[1:2],
        'base_pos_z' : base_position[2:],
        'base_pos' : base_position,
        'base_ori_euler' : base_orientation_euler,
        'base_ori_quat'  : base_orientation_quat,
        'base_velocity' : base_velocity,
        'base_ang_vel' : base_angular_velocity,
        'ori_matrix': base_orientation_matrix,
        'j_pos' : joint_positions,
        'j_vel' : joint_velocities,
        'j_eff' : joint_effort,
        'foot_pos': np.reshape(foot_pos, 12),
        'foot_pos_robot': np.reshape(foot_pos-base_position,12) # for testing IK
        }


    def set_friction_and_restitution(self, bullet_client, lateralFriction=0.5, restitution=0.1, linearDamping = 0.0, angularDamping = 0.0,):
        body_idx = self.robot_body.bodies[0]
        numLinks = bullet_client.getNumJoints(body_idx)
        bullet_client.changeDynamics(body_idx, -1, restitution=restitution, lateralFriction=lateralFriction,
                                linearDamping=linearDamping, angularDamping=angularDamping)
        # print("Set restitution {} and friction {} for body/robot {}".format(restitution, lateralFriction, body_idx))

        for joint_idx in range(numLinks):
            bullet_client.changeDynamics(body_idx, joint_idx, restitution=restitution, lateralFriction=lateralFriction, 
                                        linearDamping=linearDamping, angularDamping=angularDamping)
