from overcooked import World

def test_reward1():
    world = World(2, recipe_filename='assets/recipe.json', use_state_observation=False, level='level_0')
    world.reset()
    

    obs, success, info = world.step(['goto_agent0_storage0', 'goto_agent1_storage0'])
  
    world.step(['get_agent0_flour_storage0', 'get_agent1_pork_storage0'])
  
    world.step(['goto_agent0_blender0', 'goto_agent1_blender0'])
  
    obs, success, info = world.step(['put_agent0_flour_blender0', 'put_agent1_pork_blender0'])
    reward = info['reward']
    # assert reward == 0.5 + 0.01 - 0.05
    obs, success, info = world.step(['activate_agent0_blender0', 'noop_agent1'])

def test_reward2():
    world = World(2, recipe_filename='assets/recipe.json', use_state_observation=False, level='level_11', use_task_lifetime_interval_oracle=True)
    world.reset()
    

    obs, success, info = world.step(['goto_agent0_storage0', 'goto_agent1_storage0'])
    print(info['reward'])
  
    obs, success, info = world.step(['get_agent0_tomato_storage0', 'get_agent1_pasta_storage0'])
    print(info['reward'])
  
    obs, success, info = world.step(['goto_agent0_pan0', 'goto_agent1_pot0'])
    print(info['reward'])
  
    obs, success, info = world.step(['put_agent0_pan0', 'put_agent1_pot0'])
    print(info['reward'])

    # assert reward == 0.5 + 0.01 - 0.05
    obs, success, info = world.step(['activate_agent0_pan0', 'activate_agent1_pot0'])
    print(info['reward'])

    obs, success, info = world.step(['noop_agent0', 'noop_agent1'])
    print(info['reward'])

    obs, success, info = world.step(['noop_agent0', 'noop_agent1'])
    print(info['reward'])

    obs, success, info = world.step(['noop_agent0', 'noop_agent1'])
    print(info['reward'])

    obs, success, info = world.step(['noop_agent0', 'get_agent1_pot0'])
    # print(world.agents[1].holding.name)
    print(info['reward'])

    obs, success, info = world.step(['noop_agent0', 'goto_agent1_pan0'])
    print(info['reward'])

    obs, success, info = world.step(['noop_agent0', 'put_agent1_pan0'])
    print(info['reward'])

    obs, success, info = world.step(['activate_agent0_pan0', 'noop_agent1'])
    print(info['reward'])

    obs, success, info = world.step(['noop_agent0', 'noop_agent1'])
    print(info['reward'])
    obs, success, info = world.step(['noop_agent0', 'noop_agent1'])
    print(info['reward'])
    obs, success, info = world.step(['noop_agent0', 'get_agent1_pan0'])
    print(info['reward'])

    obs, success, info = world.step(['noop_agent0', 'goto_agent1_servingtable0'])
    print(info['reward'])

    obs, success, info = world.step(['noop_agent0', 'put_agent1_servingtable0'])
    print(info['reward'])
