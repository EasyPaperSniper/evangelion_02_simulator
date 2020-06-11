import time

import numpy as np
import pybullet

from pybullet_envs.env_bases import MJCFBaseBulletEnv
from .unitree_pybullet.StadiumScene import SinglePlayerStadiumScene


class UnitreeBase(MJCFBaseBulletEnv):
    def __init__(self, robot,
        render=False,
        sim_timestep = 1./1000.,
        sim_frameskip = 10,
        sim_numSolverIterations = 10,
        do_hard_reset = False,
        lateralFriction_plane=0.8,
        spinningFriction_plane=0.1, 
        rollingFriction_plane = 0.1,
        restitution_plane=0.0,
        COV_ENABLE_PLANAR_REFLECTION_plane= 0,
        COV_ENABLE_SHADOWS = False,
        **kwargs
        ):
        """
        Params:
            robot (object): Robot object
            render (bool): If True then a GUI will be opened. Slows down training
                process, though.
            sim_timestep (float): Time of a single step in simulation
            sim_frameskip (int): Number of sub-steps per step in simultion. Each one
                should take sim_timestep/sim_frameskip seconds.
                Higher numbers might lead to higher accuracy but takes more time.
            sim_numSolverIterations (int): Max number of iterations of the solver in
                simulation. The higher the more accurate but also slower.
            do_hard_reset (bool): If true then the urdf will be reloaded for each
                reset. Otherwise we try to reset the simulator state.
        """
        super().__init__(robot, render)
        self._p = None
        # parama
        self._param_sim_timestep = sim_timestep
        self._param_sim_frameskip = sim_frameskip
        self._param_do_hard_reset = do_hard_reset
        self._param_sim_numSolverIterations = sim_numSolverIterations
        self._param_init_camera_width = 320
        self._param_init_camera_height = 200
        self._param_lateralFriction_plane = lateralFriction_plane
        self._param_spinningFriction_plane = spinningFriction_plane
        self._param_rollingFriction_plane = rollingFriction_plane
        self._param_restitution_plane=restitution_plane
        self._param_COV_ENABLE_PLANAR_REFLECTION_plane= COV_ENABLE_PLANAR_REFLECTION_plane
        self._param_COV_ENABLE_SHADOWS = COV_ENABLE_SHADOWS
        
        self.stateId=-1
        self._set_sim_param = False
        self.camera_x = None
        self._environment_hooks = []
        self._projectM = None

        self._do_render = render
        self._do_render_camera_image = True

        state = self._robot_base_reset()
        self._observation_spec = {}
        for key, val in state.items():
            self._observation_spec[key] = (len(val),)
        self._action_spec = (len(self.robot.ordered_joints),)

    
    def create_single_player_scene(self, bullet_client):
        self.stadium_scene = SinglePlayerStadiumScene(
            bullet_client=bullet_client,
            gravity=9.8,
            timestep=self._param_sim_timestep,
            frame_skip=self._param_sim_frameskip,
            lateralFriction=self._param_lateralFriction_plane,
            spinningFriction = self._param_spinningFriction_plane,
            rollingFriction = self._param_rollingFriction_plane,
            restitution=self._param_restitution_plane,
            COV_ENABLE_PLANAR_REFLECTION_plane=self._param_COV_ENABLE_PLANAR_REFLECTION_plane,
            )
        return self.stadium_scene

    def get_observation_spec(self):
        """Returns dict of tubles defining the size of the states.

        Returns:
            dict: Size of the states.
        """
        return self._observation_spec

    def get_action_spec(self):
        """Returns the size/length of the expected actions.

        Returns:
            tuple: action length
        """
        return self._action_spec

    def reset(self):
        """Resets the simulator.

        Returns:
            state (dict): The dict contains four different states. 'j_pos' are the
                joint positions. 'j_vel' are the current velocities of the
                joint angles. 'base_pos' is the 3D position of the base of Daisy
                in the world. 'base_ori_euler' is the orientation of the robot
                in euler angles.
            reward (float): Reward achieved.
            done (bool): True if end of simulation reached. (Currently always False)
        """
        return self._robot_base_reset()

    def _robot_base_reset(self):
        """Resets the simulator.

        Returns:
            state (dict): The dict contains four different states. 'j_pos' are the
                joint positions. 'j_vel' are the current velocities of the
                joint angles. 'base_pos' is the 3D position of the base of Daisy
                in the world. 'base_ori_euler' is the orientation of the robot
                in euler angles.
            reward (float): Reward achieved.
            done (bool): True if end of simulation reached. (Currently always False)
        """
        self._do_reset_simulator_preparations()
        if (self.stateId>=0):
            self._p.restoreState(self.stateId)

        if self._p is not None and self._do_render:
            self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING,0)
        state = super(UnitreeBase, self).reset()
        state['j_eff'] = [0.0 for _ in state['j_eff']]
        # state['j_eff'] contains the efforts produced from the last time step
        # We have to reset it here to zero or otherwise we have the wrong information

        # if not self._set_sim_param:
        #     # self._p.setPhysicsEngineParameter(enableFileCaching=0)
        #     self._set_sim_param = True

        # self.parts, self.jdict, self.ordered_joints, self.robot_body = self.robot.addToScene(self._p,
        #     self.stadium_scene.ground_plane_mjcf)

        if self._do_render:
            self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING,1)
            # self._p.configureDebugVisualizer(pybullet.COV_ENABLE_GUI,1)

        if self._param_COV_ENABLE_SHADOWS:
            self._p.configureDebugVisualizer(pybullet.COV_ENABLE_SHADOWS, 1)
        if (self.stateId<0):
            self.stateId=self._p.saveState()
            print("saving state self.stateId:",self.stateId)


        self.camera_x = None
        self.camera_adjust()

        return state

    def step(self, a):
        """Applies action a and returns next state.

        Args:
            a (list or numpy array): List of actions. Lenght must be the same as
                get_action_spec() defines.

        Returns:
            state (dict): The dict contains four different states. 'j_pos' are the
                joint positions. 'j_vel' are the current velocities of the
                joint angles. 'base_pos' is the 3D position of the base of Daisy
                in the world. 'base_ori_euler' is the orientation of the robot
                in euler angles.
            reward (float): Reward achieved.
            done (bool): True if end of simulation reached. (Currently always False)

        """
        # Avoids the camera reset by the parent class
        return self._step(a)

    def _step(self, a):
        """Applies action a and returns next state.

        Args:
            a (list or numpy array): List of actions. Lenght must be the same as
                get_action_spec() defines.

        Returns:
            state (dict): The dict contains four different states. 'j_pos' are the
                joint positions. 'j_vel' are the current velocities of the
                joint angles. 'base_pos' is the 3D position of the base of Daisy
                in the world. 'base_ori_euler' is the orientation of the robot
                in euler angles.
            reward (float): Reward achieved.
            done (bool): True if end of simulation reached. (Currently always False)

        """
        if isinstance(a, np.ndarray):
            a = a.tolist()
        self.robot.apply_action(a)
        self.scene.global_step()
        self.camera_adjust()
        if self._do_render:
            self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING,1)
        state = self.robot.calc_state()

        return state

    def camera_adjust(self):
        if self._p is None or not self._do_render:
            return
        self.camera._p = self._p
        x, y, z = self.robot.body_xyz
        if self.camera_x is not None:
            self.camera_x = x # 0.98*self.camera_x + (1-0.98)*x
        else:
            self.camera_x = x
        # self.camera.move_and_look_at(self.camera_x, y-2.0, 1.4, x, y, 1.0)
        lookat = [self.camera_x, y, z]
        distance = 2.0
        yaw = 10
        pitch = -20
        self._p.resetDebugVisualizerCamera(distance, yaw, pitch, lookat)

    
    def render_camera_image(self, pixelWidthHeight = None):
        if pixelWidthHeight is not None or self._projectM is None:
            if self._projectM is None:
                self._pixelWidth = self._param_init_camera_width
                self._pixelHeight = self._param_init_camera_height
            else:
                self._pixelWidth = pixelWidthHeight[0]
                self._pixelHeight = pixelWidthHeight[1]
            nearPlane = 0.01
            farPlane = 10
            aspect = self._pixelWidth / self._pixelHeight
            fov = 60
            self._projectM = self._p.computeProjectionMatrixFOV(fov, aspect,
                nearPlane, farPlane)

        x, y, z = self.robot.body_xyz
        lookat = [x, y, z]
        distance = 1.4
        pitch = 90
        yaw = 0
        roll = 0
        viewM = self._p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=lookat,
            distance=distance,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            upAxisIndex=2)

        # img_arr, depth_arr, mask_arr = pybullet.getCameraImage(self._pixelWidth, self._pixelHeight, viewM, self._projectM, shadow=0,renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
        img_arr = pybullet.getCameraImage(self._pixelWidth, self._pixelHeight, viewM, self._projectM, shadow=0, renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)#, flags=pybullet.COV_ENABLE_SEGMENTATION_MARK_PREVIEW)

        w=img_arr[0] #width of the image, in pixels
        h=img_arr[1] #height of the image, in pixels
        rgb=img_arr[2] #color data RGB

        image = np.reshape(rgb, (h, w, 4)) #Red, Green, Blue, Alpha
        image = image * (1./255.)
        image = image[:,:,0:3]
        return image #, depth_arr, mask_arr

    
    def _do_reset_simulator_preparations(self, reload_scene=False):
        if self._param_do_hard_reset:
            self.stateId = -1
            if self._p is not None:
                self._p.resetSimulation()
            reload_scene = True

        if self.scene is not None:
            self.scene.cpp_world.numSolverIterations =  self._param_sim_numSolverIterations
            if reload_scene:
                self.scene.stadiumLoaded = 0