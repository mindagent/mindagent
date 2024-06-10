import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from overcooked import World
from overcooked.wrappers.ar_nn_wrapper import ARNNWrapper
import gymnasium as gym
import gymnasium.spaces as spaces
# def test_1():
#     world = World(2, recipe_filename='assets/recipe.json', task_filename='test/assets/tasks.json')
    
#     env : ARNNWrapper = ARNNWrapper(sim=world)
#     obs1 = env.reset(task_name='tomatoPasta')

#     action = env.action_space.no_op()

#     obs2, reward, success, info = env.step(action)

#     def remove_first_five_lines(text):
#         lines = text.split('\n')
#         remaining_lines = lines[5:]
#         return '\n'.join(remaining_lines)

#     assert remove_first_five_lines(obs1['prompt']) == remove_first_five_lines(obs2['prompt'])

#     assert reward == -0.05 


def test_2():
    world = World(2, task_lifetime = 20, recipe_filename='assets/recipe.json', task_filename='test/assets/tasks.json')
    
    env : ARNNWrapper = ARNNWrapper(sim=world)
    env.reset(options={'task_name': 'porkMeatcake'})

    ra = env.reverse_action(['goto_agent0_storage0', 'goto_agent1_storage0'])
    obs, reward, done, truncated, info = env.step(ra)
    
    ra = env.reverse_action(['get_agent0_flour_storage0', 'get_agent1_pork_storage0'])
    obs, reward, done, truncated, info = env.step(ra)

    ra = env.reverse_action(['goto_agent0_blender0', 'goto_agent1_blender0'])
    obs, reward, done, truncated, info = env.step(ra)

    ra = env.reverse_action(['put_agent0_blender0', 'put_agent1_blender0'])
    obs, reward, done, truncated, info = env.step(ra)

    ra = env.reverse_action(['activate_agent0_blender0', 'noop_agent1'])
    obs, reward, done, truncated, info = env.step(ra)

    ra = env.reverse_action(['noop_agent0', 'noop_agent1'])
    obs, reward, done, truncated, info = env.step(ra)
    ra = env.reverse_action(['noop_agent0', 'noop_agent1'])
    obs, reward, done, truncated, info = env.step(ra)

    ra = env.reverse_action(['get_agent0_blender0', 'noop_agent1'])
    obs, reward, done, truncated, info = env.step(ra)

    ra = env.reverse_action(['goto_agent0_servingtable0', 'noop_agent1'])
    obs, reward, done, truncated, info = env.step(ra)

    ra = env.reverse_action(['put_agent0_servingtable0', 'noop_agent1'])
    obs, reward, done, truncated, info = env.step(ra)
    # assert reward == 0.96
    assert info['just_success'] == True

def test3():
    from overcooked.wrappers.dummpy_vec_wrapper import DummyVecEnv
    
    def create_environment():
        world = World(2, recipe_filename='assets/recipe.json', task_filename='test/assets/tasks.json')
        return ARNNWrapper(sim=world)
    
    num_envs = 2
    env_lst = [create_environment for _ in range(num_envs)]

    env = DummyVecEnv(env_lst)
    env.reset()
    actions = []
    action =   ['goto_agent0_storage0', 'goto_agent1_storage0']
    for _ in range(num_envs):
        actions.append(action)

    ras = env.reverse_action(actions)
    obs, reward, dones, infos = env.step(ras)

# def test_4():
#     world = World(2, recipe_filename='assets/recipe.json', task_filename='test/assets/tasks.json')
    
#     env : ARNNWrapper = ARNNWrapper(sim=world)
#     env.reset(task_name='porkMeatcake')

#     ra = env.reverse_action(['goto_agent0_storage0', 'goto_agent1_storage0'])
#     obs, reward, success, info = env.step(ra)
    
#     from sentence_transformers import SentenceTransformer
#     model = SentenceTransformer('all-MiniLM-L6-v2')
#     sentence = obs['prompt']
#     embeddings = model.encode(sentence)

def test_mask():
    world = World(2, recipe_filename='assets/recipe.json', task_filename='test/assets/tasks.json')
    
    env : ARNNWrapper = ARNNWrapper(sim=world)
    env.reset(options={'task_name': 'porkMeatcake'})
    ra = env.reverse_action(['goto_agent0_storage0', 'goto_agent1_storage0'])
    obs, reward, success, truncated, info = env.step(ra)
    # import numpy as np
    # print(env.action_space)
    # print(env.get_mask())



    



