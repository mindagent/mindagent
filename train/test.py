
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import PPO
import gymnasium as gym
import numpy as np

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

import os
import sys
import gym
import numpy as np
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from feature_extractor.sbert import SBert
from overcooked import World
from overcooked.wrappers.ar_nn_wrapper import ARNNWrapper
    
def create_environment():
    world = World(2, recipe_filename='assets/recipe.json', use_state_observation=False, 
                  use_task_lifetime_interval_oracle=True,level='level_2')
    return ARNNWrapper(sim=world)

def mask_fn(env: gym.Env) -> np.ndarray:
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    return env.get_mask()

num_envs = 1



env = create_environment()
env = ActionMasker(env, mask_fn)


model = MaskablePPO.load("checkpoints/ppo_mask_level_test2_long_train_new_reward_11300000_steps", 
                         env)


vec_env = model.get_env()
cnt_pork = 0
success_cnt_total = []
fail_cnt_total = []
total_success = []
total = []
from collections import Counter
for _ in tqdm(range(1000)):
    obs = vec_env.reset()
    total_rewards = 0
    # for i in range(60):
    #     action, _states = model.predict(obs, deterministic=True)
        
    #     # print(env.action(action[0]))
    #     obs, rewards, dones, info = vec_env.step(action)
    #     # print('step reward', rewards)
    #     total_rewards += rewards
    for i in range(60):
        # Retrieve current action mask
        action_masks = get_action_masks(env)
        action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
        if len(action) == 1:
            action = action[0]
        # print(env.action(action))
        obs, reward, terminated, truncated, info = env.step(action)
        # print(reward)
        total_rewards += reward

    # print('total reward: ', total_rewards)
    # print(env.task_manager.accomplished_tasks())
    # print("failed_cnt: ", env.failed_count)
    # print('success count: ', env.success_count)
    total_success.extend(env.task_manager.accomplished_tasks())
    success_cnt_total.append(env.success_count)
    fail_cnt_total.append(env.failed_count)
    total.append(env.success_count + env.failed_count )

print('success cnt total: ', np.mean(success_cnt_total))
print('failed cnt total: ',  np.mean(fail_cnt_total))
print('max success: ', np.max(success_cnt_total))
print('total: ', total)
counter = Counter(total_success)
for element, count in counter.items():
    print(f"{element}: {count}")

    # print()
    # if env.task_manager.accomplished_tasks()[0] ==  'porkMeatcake':
        # print("pork first")
        # cnt_pork += 1
   
    # vec_env.render("human")
    # print(rewards)
# import gym

# from stable_baselines3 import PPO
# from stable_baselines3.common.env_util import make_vec_env

# # Parallel environments
# vec_env = make_vec_env("CartPole-v1", n_envs=4)

# model = PPO("MlpPolicy", vec_env, verbose=1)
# model.learn(total_timesteps=25000)