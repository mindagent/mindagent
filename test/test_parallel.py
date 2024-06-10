import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from overcooked import World

# Note: No action is executed in parallel, we must explictly specify the order of actions.

# put and activate conflict with an empty tool
def test_parallel2():
    world = World(2, recipe_filename='assets/recipe.json')
    world.reset(task_name='beefMeatcake')

    obs, success, info = world.step(['goto_agent0_storage0', 'goto_agent1_storage0'])
    obs, success, info = world.step(['get_agent0_beef_storage0', 'goto_agent1_blender0'])
    obs, success, info = world.step(['goto_agent0_blender0', 'goto_agent1_blender0'])
    obs, success, info = world.step(['put_agent0_beef_blender0','activate_agent1_blender0'])

    assert world.blender0.content.name == 'groundBeef'
    assert world.agents[0].holding is None
    assert world.agents[1].holding is None

# put and activate conflict with an empty tool
def test_parallel3():
    world = World(2, recipe_filename='assets/recipe.json')
    world.reset(task_name='beefMeatcake')

    obs, success, info = world.step(['goto_agent0_storage0', 'goto_agent1_storage0'])
    obs, success, info = world.step(['get_agent0_beef_storage0', 'goto_agent1_blender0'])
    obs, success, info = world.step(['goto_agent0_blender0', 'goto_agent1_blender0'])
    # activate will be ignore as the tool is empty
    obs, success, info = world.step(['activate_agent1_blender0', 'put_agent0_beef_blender0'])

    assert world.blender0.content.name == 'beef'
    assert world.agents[0].holding is None
    assert world.agents[1].holding is None

# put and activate conflict with a non-empty tool
def test_parallel4():
    world = World(2, recipe_filename='assets/recipe.json')
    world.reset(task_name='beefMeatcake')

    obs, success, info = world.step(['goto_agent0_storage0', 'goto_agent1_storage0'])
    obs, success, info = world.step(['get_agent0_beef_storage0', 'get_agent1_flour_storage0'])
    obs, success, info = world.step(['goto_agent0_blender0', 'goto_agent1_blender0'])
    obs, success, info = world.step(['put_agent0_beef_blender0', 'noop_agent1'])
    # once the blender is activated, the put action will fail
    obs, success, info = world.step(['activate_agent0_blender0', 'put_agent1_flour_blender0'])

    assert world.blender0.content.name == 'groundBeef'
    assert world.agents[0].holding is None
    assert world.agents[1].holding.name == 'flour'

# put and activate conflict with a non-empty tool
def test_parallel5():
    world = World(2, recipe_filename='assets/recipe.json')
    world.reset(task_name='beefMeatcake')

    obs, success, info = world.step(['goto_agent0_storage0', 'goto_agent1_storage0'])
    obs, success, info = world.step(['get_agent0_beef_storage0', 'get_agent1_flour_storage0'])
    obs, success, info = world.step(['goto_agent0_blender0', 'goto_agent1_blender0'])
    obs, success, info = world.step(['put_agent0_beef_blender0', 'noop_agent1'])
    obs, success, info = world.step(['put_agent1_flour_blender0', 'activate_agent0_blender0'])

    assert world.blender0.content.name == 'beefMeatcake'
    assert world.agents[0].holding is None
    assert world.agents[1].holding is None

# put and get conflict with an empty tool
def test_parallel6():
    world = World(2, recipe_filename='assets/recipe.json')
    world.reset(task_name='beefMeatcake')

    obs, success, info = world.step(['goto_agent0_storage0', 'goto_agent1_storage0'])
    obs, success, info = world.step(['get_agent0_beef_storage0', 'noop_agent1'])
    obs, success, info = world.step(['goto_agent0_blender0', 'goto_agent1_blender0'])
    obs, success, info = world.step(['put_agent0_beef_blender0', 'get_agent1_beef_blender0'])

    assert world.blender0.content is None
    assert world.agents[0].holding is None
    assert world.agents[1].holding.name == 'beef'

# put and get conflict with an empty tool
def test_parallel7():
    world = World(2, recipe_filename='assets/recipe.json')
    world.reset(task_name='beefMeatcake')

    obs, success, info = world.step(['goto_agent0_storage0', 'goto_agent1_storage0'])
    obs, success, info = world.step(['get_agent0_beef_storage0', 'noop_agent1'])
    obs, success, info = world.step(['goto_agent0_blender0', 'goto_agent1_blender0'])
    obs, success, info = world.step(['get_agent1_beef_blender0', 'put_agent0_beef_blender0'])

    assert world.blender0.content.name == 'beef'
    assert world.agents[0].holding is None
    assert world.agents[1].holding is None

# put and get conflict with an non-empty tool
def test_parallel8():
    world = World(2, recipe_filename='assets/recipe.json')
    world.reset(task_name='beefMeatcake')

    obs, success, info = world.step(['goto_agent0_blender0', 'goto_agent1_storage0'])
    obs, success, info = world.step(['noop_agent0', 'get_agent1_beef_storage0'])
    obs, success, info = world.step(['noop_agent0', 'goto_agent1_blender0'])
    obs, success, info = world.step(['noop_agent0', 'put_agent1_beef_blender0'])

    assert world.blender0.content.name == 'beef'
    assert world.agents[0].holding is None
    assert world.agents[1].holding is None

    obs, success, info = world.step(['noop_agent0', 'goto_agent1_storage0'])
    obs, success, info = world.step(['noop_agent0', 'get_agent1_flour_storage0'])
    obs, success, info = world.step(['noop_agent0', 'goto_agent1_blender0'])
    obs, success, info = world.step(['get_agent0_beef_blender0', 'put_agent1_flour_blender0'])

    assert world.blender0.content.name == 'flour'
    assert world.agents[0].holding.name == 'beef'
    assert world.agents[1].holding is None
