import time
import os

import numpy as np

import unitree_env as e


env = e.UnitreeBasicEnv(render=True, robot ='a1',control_mode='position')

state = env.reset()
for _ in range(200):
    action = state['j_pos']
    state_ = env.step(action)


env.reset()
for _ in range(200):
    action = np.random.randn(12)
    state_ = env.step(action)