from typing import Union, Optional,Any

import gymnasium as gym
import gymnasium.spaces as spaces

from .masks_wrapper import ARMasksWrapper as _ARMasksWrapper
from .nn_action_space_wrapper import NNActionSpaceWrapper as _NNActionSpaceWrapper
from stable_baselines3.common.monitor import Monitor
class ARNNWrapper(gym.Wrapper):
    def __init__(
        self,
        sim,
    ):
        sim = Monitor(_ARMasksWrapper(
                _NNActionSpaceWrapper(
                    env=sim,
                ),
            ))
        super().__init__(env=sim)

    def get_mask(self):
        return self.env.get_mask()
    
    def action(self, action):
        return self.env.env.action(action)

    def reverse_action(self, action):
        return self.env.env.reverse_action(action)

