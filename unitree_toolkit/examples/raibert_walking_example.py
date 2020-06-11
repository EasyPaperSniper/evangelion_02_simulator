import numpy as np
import matplotlib.pyplot as plt

from unitree_toolkit.unitree_utils.raibert_walking_controller import Raibert_controller
from unitree_toolkit.unitree_robot_env import unitree_robot_API

env = unitree_robot_API(render=True, robot ='a1',control_mode='position')
raibert_controller = Raibert_controller()

target_speed = np.array([0.0,0.05,0])

state = env.reset()
for _ in range(10):
    action = state['j_pos']
    state = env.step(action)
raibert_controller.set_init_state(state)



for _ in range(10):
    latent_action = raibert_controller.plan_latent_action(state, target_speed)
    raibert_controller.update_latent_action(state, latent_action)
    for timestep in range(1, 100+1):
        action = raibert_controller.get_action(state, timestep)
        state = env.step(action)


