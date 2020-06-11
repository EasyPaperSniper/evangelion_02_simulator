import time

import numpy as np

from .unitree_robot_base import UnitreeRobot
from .unitree_env_base import UnitreeBase


class UnitreeBasicEnv(UnitreeBase):
    def __init__(self, robot='a1', **kwargs):
        if 'restitution' in kwargs.keys():
            sqrt_restitution = np.sqrt(kwargs['restitution'])
            kwargs['restitution_plane'] = sqrt_restitution
            kwargs['restitution_robot'] = sqrt_restitution

        if 'lateralFriction' in kwargs.keys():
            sqrt_lateralFriction = np.sqrt(kwargs['lateralFriction'])
            kwargs['lateralFriction_plane'] = sqrt_lateralFriction
            kwargs['lateralFriction_robot'] = sqrt_lateralFriction

        self.robot = UnitreeRobot(robot_type=robot, **kwargs)
        super().__init__(self.robot, **kwargs)