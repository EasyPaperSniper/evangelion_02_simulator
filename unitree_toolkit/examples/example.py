import time
import os

import numpy as np

from unitree_toolkit.unitree_robot_env import unitree_robot_API

env = unitree_robot_API(render=True, robot ='a1',control_mode='position')


state = env.reset()
for _ in range(200):
    action = state['j_pos']
    state_ = env.step(action)


env.reset()
for _ in range(200):
    action = np.random.randn(12)
    state_ = env.step(action)