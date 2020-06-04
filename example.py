import time
import os

import numpy as np

import unitree_env as e


env = e.UnitreeBasicEnv(render=True, robot ='a1',control_mode='position')
state = env.reset()

for _ in range(1000):
    action = np.random.randn(12)
    # action[8] = -2
    state_ = env.step(action)


# while(1):
#     with open("mocap.txt","r") as filestream:
#         for line in filestream:
#             currentline = line.split(",")
#             frame = currentline[0]
#             t = currentline[1]
#             joints=currentline[2:14]
#             action = []
#             for i in range(12):
#                 action.append(float(joints[i]))
#             env.step(action)