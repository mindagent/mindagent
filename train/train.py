from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import PPO
import gymnasium as gym
import numpy as np
import argparse

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib import RecurrentPPO
from stable_baselines3 import SAC

import os
import sys
import gym
import numpy as np
import torch as th
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from feature_extractor.sbert import SBert
from overcooked import World
from overcooked.wrappers.ar_nn_wrapper import ARNNWrapper

# policy_kwargs = dict(
#     features_extractor_class=SBert,
#     # features_extractor_kwargs=dict(features_dim=128),
# )

from overcooked.wrappers.dummpy_vec_wrapper import DummyVecEnv
  
def main(level):
  def create_environment():
      world = World(2, recipe_filename='assets/recipe.json', use_state_observation=False, level=level,
                    use_task_lifetime_interval_oracle=True)
      return ARNNWrapper(sim=world)

  def mask_fn(env: gym.Env) -> np.ndarray:
      # Do whatever you'd like in this function to return the action mask
      # for the current env. In this example, we assume the env has a
      # helpful method we can rely on.
      return env.get_mask()

  checkpoint_callback = CheckpointCallback(
    save_freq=100000,
    save_path="./checkpoints/",
    name_prefix=f"ppo_mask_{level}_long_train_new_reward",
    save_replay_buffer=True,
    save_vecnormalize=True,
  )

  policy_kwargs = dict(activation_fn=th.nn.ReLU,
                      net_arch=dict(pi=[512, 512, 512], vf=[512, 512, 512]))

# env_lst = [create_environment for _ in range(num_envs)]

  env = create_environment()
  env = ActionMasker(env, mask_fn)
  # env = DummyVecEnv(env_lst)

  # model = RecurrentPPO("MlpLstmPolicy", env, verbose=1, tensorboard_log="./run" )

  model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1, 
                      policy_kwargs=policy_kwargs, tensorboard_log="./run",
                      n_steps = 2400, ent_coef= 0.03)
  # model  = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs)
  model.learn(50000000, callback=checkpoint_callback, tb_log_name="run_bert_large")
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train RL using sb3 contrib masked ppo')
    parser.add_argument('--level', help='what level to train')

    args = parser.parse_args()

    main(args.level)