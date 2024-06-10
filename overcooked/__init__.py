from .game import World
import gym
gym.envs.register(
     id='Overcooked-v0',
     entry_point='overcooked:ARNNWrapper',
     max_episode_steps=60,
)