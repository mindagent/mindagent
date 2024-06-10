import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from overcooked import World


# blender does not allow to pick up anything from mixup
def test_tool_pickup_1():
    world = World(2, recipe_filename='assets/recipe.json')
    world.reset(task_name='beefMeatcake')

    obs, success, info = world.step(['goto_agent0_blender0', 'goto_agent1_storage0'])
    obs, success, info = world.step(['noop_agent0', 'get_agent1_beef_storage0'])
    obs, success, info = world.step(['noop_agent0', 'goto_agent1_blender0'])
    obs, success, info = world.step(['noop_agent0', 'put_agent1_beef_blender0'])

    assert world.blender0.content.name == 'beef'
    assert world.agents[0].holding is None
    assert world.agents[1].holding is None

    obs, success, info = world.step(['noop_agent0', 'get_agent1_beef_blender0'])

    assert world.blender0.content is None
    assert world.agents[0].holding is None
    assert world.agents[1].holding.name == 'beef'

    obs, success, info = world.step(['noop_agent0', 'put_agent1_beef_blender0'])

    obs, success, info = world.step(['noop_agent0', 'goto_agent1_storage0'])
    obs, success, info = world.step(['noop_agent0', 'get_agent1_flour_storage0'])
    obs, success, info = world.step(['noop_agent0', 'goto_agent1_blender0'])
    obs, success, info = world.step(['put_agent1_flour_blender0', 'get_agent0_beef_blender0'])

    assert world.blender0.content is None
    assert world.agents[1].holding is None
    assert world.agents[0].holding.name == 'waste'
