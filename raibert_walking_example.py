from unitree_utils.raibert_walking_controller import Raibert_controller
import unitree_env as e
import numpy as np


env = e.UnitreeBasicEnv(render=True, robot ='a1',control_mode='position')
raibert_controller = Raibert_controller()

state = env.reset()
for _ in range(50):
    action = state['j_pos']
    state_ = env.step(action)

raibert_controller.set_init_state(state_)
for _ in range(1000):
    latent_action = raibert_controller.plan_latent_action(state, np.array([0.1,0,0]))
    raibert_controller.update_latent_action(state, latent_action)
    for timestep in range(1, 50+1):
        action = raibert_controller.get_action(state, timestep)
        state = env.step(action)