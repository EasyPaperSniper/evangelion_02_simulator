import numpy as np

class unitree_robot_API():
    def __init__(self, robot = 'a1', sim=True, render=False,  logger=False, control_mode='position'):
        self.sim = sim
        if self.sim:
            import unitree_sim_env as e
            self.env = e.UnitreeBasicEnv(render=render, robot = robot ,control_mode=control_mode)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)