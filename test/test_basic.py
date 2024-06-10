"""
pytest
pytest -rP
"""
import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from overcooked import World


def test_0():
    world = World(3, recipe_filename='assets/recipe.json')
    world.reset(task_name='beefMeatcake')
    world.reset(task_name='porkMeatcake')
    obs, success, info = world.step(['goto'])
    assert all(x == y for x, y in zip(info['action_success'], [False]))

    obs, success, info = world.step(['goto_agent1_storage0_pan0'])
    assert all(x == y for x, y in zip(info['action_success'], [False]))

    obs, success, info = world.step(['test_agent0'])
    assert all(x == y for x, y in zip(info['action_success'], [False]))

    obs, success, info = world.step(['goto_player0_storage0'])
    assert all(x == y for x, y in zip(info['action_success'], [False]))

    obs, success, info = world.step(['goto_agent0_unknownlocation'])
    assert all(x == y for x, y in zip(info['action_success'], [False]))

    obs, success, info = world.step(['goto_agent1_storage0', 'goto_agent1_servingtable0'])
    assert all(x == y for x, y in zip(info['action_success'], [False, False]))

def test_get1():
    world = World(3, recipe_filename='assets/recipe.json')
    world.reset(task_name='beefMeatcake')
    obs, success, info = world.step(['get_agent0_tomato_storage0'])
    assert all(x == y for x, y in zip(info['action_success'], [False]))

    obs, success, info = world.step(['get_agent0_tomato_blender0'])
    assert all(x == y for x, y in zip(info['action_success'], [False]))


    obs, success, info = world.step(['get_agent0_tomato_storage0'])
    assert all(x == y for x, y in zip(info['action_success'], [False]))
    obs, success, info = world.step(['goto_agent0_blender0'])
    obs, success, info = world.step(['get_agent0_tomato_blender0'])
    assert all(x == y for x, y in zip(info['action_success'], [False]))

def test_get2():
    world = World(3, recipe_filename='assets/recipe.json')
    world.reset(task_name='beefMeatcake')
    obs, success, info = world.step(['goto_agent0_storage0'])
    assert all(x == y for x, y in zip(info['action_success'], [True]))
    obs, success, info = world.step(['get_agent0_beef_storage0'])
    obs, success, info = world.step(['goto_agent0_blender0'])
    obs, success, info = world.step(['put_agent0_beef_blender0'])
    assert all(x == y for x, y in zip(info['action_success'], [True]))
    obs, success, info = world.step(['get_agent0_beef_blender0'])
    # assert success == False
    assert all(x == y for x, y in zip(info['action_success'], [True]))
    assert world.agents[0].holding.name == 'beef'

def test_put1():
    world = World(3, recipe_filename='assets/recipe.json')
    world.reset(task_name='beefMeatcake')
    obs, success, info = world.step(['put_agent0_beef_storage0'])
    assert all(x == y for x, y in zip(info['action_success'], [False]))

    obs, success, info = world.step(['goto_agent0_storage0'])

    obs, success, info = world.step(['put_agent0_beef_storage0'])
    assert all(x == y for x, y in zip(info['action_success'], [False]))
    obs, success, info = world.step(['get_agent0_beef_storage0'])
    assert all(x == y for x, y in zip(info['action_success'], [True]))
    assert world.agents[0].holding.name == 'beef'
    obs, success, info = world.step(['put_agent0_beef_storage0'])
    assert world.agents[0].holding is None

    obs, success, info = world.step(['get_agent0_beef_storage0'])
    assert world.agents[0].holding.name == 'beef'
    obs, success, info = world.step(['goto_agent0_blender0'])
    obs, success, info = world.step(['put_agent0_pork_storage0'])
    assert world.agents[0].holding.name == 'beef'
    assert all(x == y for x, y in zip(info['action_success'], [False]))

    obs, success, info = world.step(['put_agent0_pork_blender0'])
    assert world.agents[0].holding is None
    assert all(x == y for x, y in zip(info['action_success'], [True]))


def test_activate1():
    world = World(3, recipe_filename='assets/recipe.json', task_filename='test/assets/tasks.json')
    world.reset(task_name='tomatoPasta')

    obs, success, info = world.step(['goto_agent0_storage0'])
    obs, success, info = world.step(['get_agent0_tomato_storage0'])
    assert all(x == y for x, y in zip(info['action_success'], [True]))
    assert world.agents[0].holding.name == 'tomato'
    obs, success, info = world.step(['goto_agent0_pan0'])
    obs, success, info = world.step(['put_agent0_tomato_pan0'])
    assert world.agents[0].holding is None
    assert all(x == y for x, y in zip(info['action_success'], [True]))
    obs, success, info  = world.step(['activate_agent0_pot0'])
    assert all(x == y for x, y in zip(info['action_success'], [False]))
    assert world.pan0.content.name == 'tomato'


def test_activate2():
    world = World(3, recipe_filename='assets/recipe.json', task_filename='test/assets/tasks.json')
    world.reset(task_name='tomatoPasta')

    obs, success, info = world.step(['goto_agent0_storage0'])
    obs, success, info = world.step(['get_agent0_tomato_storage0'])
    assert all(x == y for x, y in zip(info['action_success'], [True]))
    assert world.agents[0].holding.name == 'tomato'
    obs, success, info = world.step(['goto_agent0_pan0'])
    obs, success, info = world.step(['put_agent0_tomato_pan0'])
    assert world.agents[0].holding is None
    assert all(x == y for x, y in zip(info['action_success'], [True]))

    obs, success, info = world.step(['goto_agent0_storage0'])
    obs, success, info = world.step(['get_agent0_pasta_storage0'])
    obs, success, info = world.step(['goto_agent0_pan0'])
    obs, success, info = world.step(['activate_agent0_pan0'])
    assert world.agents[0].holding.name == 'pasta'
    assert world.pan0.content.name == 'tomato'
    assert all(x == y for x, y in zip(info['action_success'], [False]))

def test_1():
    world = World(2, recipe_filename='assets/recipe.json', task_filename='test/assets/tasks.json')
    world.reset(task_name='porkMeatcake')
    obs, success, info = world.step(['goto_agent0_storage0', 'goto_agent1_storage0'])
    expected = """
    at(agent0, storage0)
    at(agent1, storage0)
    hold(agent0, None)
    hold(agent1, None)
    inside(storage0, None)
    inside(blender0, None)
    inside(chopboard0, None)
    inside(chopboard1, None)
    inside(pan0, None)
    inside(servingtable0, None)
    """

    # assert removespace(obs) == removespace(expected)

    assert world.agents[0].location == world.storage0
    assert world.agents[1].location == world.storage0
    assert world.agents[0].holding is None
    assert world.agents[1].holding is None
    world.step(['get_agent0_flour_storage0', 'get_agent1_pork_storage0'])
    assert world.agents[0].holding.name == 'flour'
    assert world.agents[1].holding.name == 'pork'
    world.step(['goto_agent0_blender0', 'goto_agent1_blender0'])
    assert world.agents[0].location == world.blender0
    assert world.agents[1].location == world.blender0
    obs, success, info = world.step(['put_agent0_flour_blender0', 'put_agent1_pork_blender0'])
    expected = """
    at(agent0, blender0)
    at(agent1, blender0)
    hold(agent0, None)
    hold(agent1, None)
    inside(storage0, None)
    inside(blender0, flour&pork)
    inside(chopboard0, None)
    inside(chopboard1, None)
    inside(pan0, None)
    inside(servingtable0, None)
    """
    # assert removespace(obs) == removespace(expected)
    assert world.agents[0].holding is None
    assert world.agents[1].holding is None
    assert world.blender0.content.name == 'flour&pork'
    obs, success, info = world.step(['activate_agent0_blender0', 'noop_agent1'])
    expected = """
    at(agent0, blender0)
    at(agent1, blender0)
    hold(agent0, None)
    hold(agent1, None)
    inside(storage0, None)
    inside(blender0, porkMeatcake)
    inside(chopboard0, None)
    inside(chopboard1, None)
    inside(pan0, None)
    inside(servingtable0, None)
    """

    # assert removespace(obs) == removespace(expected)

    assert world.agents[0].holding is None
    assert world.agents[1].holding is None
    assert world.blender0.content.name == 'porkMeatcake'
    assert success == False
    obs, success, info = world.step(['goto_agent0_storage0', 'noop_agent1'])
    assert world.agents[0].location == world.storage0
    assert world.agents[1].location == world.blender0
    assert success == False
    assert all(x == y for x, y in zip(info['action_success'], [True, True]))

    obs, success, info = world.step(['noop_agent0', 'get_agent1_porkMeatcake_blender0'])
    assert world.agents[0].location == world.storage0
    assert world.agents[1].location == world.blender0
    assert world.agents[1].holding.name == 'porkMeatcake'
    assert world.blender0.content is None
    assert success == False
    assert all(x == y for x, y in zip(info['action_success'], [True, True]))

    obs, success, info = world.step(['noop_agent0', 'goto_agent1_servingtable0'])
    assert world.agents[0].location == world.storage0
    assert world.agents[1].location == world.servingtable0
    assert world.agents[1].holding.name == 'porkMeatcake'
    assert world.blender0.content is None
    assert success == False
    obs, success, info = world.step(['noop_agent0', 'put_agent1_porkMeatcake_servingtable0'])
    assert world.agents[0].location == world.storage0
    assert world.agents[1].location == world.servingtable0
    assert world.agents[1].holding is None
    assert world.blender0.content is None
    # when the right dish is given, the serving table will be cleared
    assert world.servingtable0.content is None
    assert world.task_manager.accomplished_tasks() == ['porkMeatcake']
    assert success == True

def test_1_1():
    world = World(2, recipe_filename='assets/recipe.json', task_filename='test/assets/tasks.json')
    world.reset(task_name='tomatoPasta')
    obs, success, info = world.step(['goto_agent0_storage0', 'goto_agent1_storage0'])
    expected = """
    at(agent0, storage0)
    at(agent1, storage0)
    hold(agent0, None)
    hold(agent1, None)
    inside(storage0, None)
    inside(blender0, None)
    inside(chopboard0, None)
    inside(chopboard1, None)
    inside(pan0, None)
    inside(servingtable0, None)
    """

    # assert removespace(obs) == removespace(expected)

    assert world.agents[0].location == world.storage0
    assert world.agents[1].location == world.storage0
    assert world.agents[0].holding is None
    assert world.agents[1].holding is None
    world.step(['get_agent0_flour_storage0', 'get_agent1_pork_storage0'])
    assert world.agents[0].holding.name == 'flour'
    assert world.agents[1].holding.name == 'pork'
    world.step(['goto_agent0_blender0', 'goto_agent1_blender0'])
    assert world.agents[0].location == world.blender0
    assert world.agents[1].location == world.blender0
    obs, success, info = world.step(['put_agent0_flour_blender0', 'put_agent1_pork_blender0'])
    expected = """
    at(agent0, blender0)
    at(agent1, blender0)
    hold(agent0, None)
    hold(agent1, None)
    inside(storage0, None)
    inside(blender0, flour&pork)
    inside(chopboard0, None)
    inside(chopboard1, None)
    inside(pan0, None)
    inside(servingtable0, None)
    """
    # assert removespace(obs) == removespace(expected)
    assert world.agents[0].holding is None
    assert world.agents[1].holding is None
    assert world.blender0.content.name == 'flour&pork'
    obs, success, info = world.step(['activate_agent0_blender0', 'noop_agent1'])
    expected = """
    at(agent0, blender0)
    at(agent1, blender0)
    hold(agent0, None)
    hold(agent1, None)
    inside(storage0, None)
    inside(blender0, porkMeatcake)
    inside(chopboard0, None)
    inside(chopboard1, None)
    inside(pan0, None)
    inside(servingtable0, None)
    """

    # assert removespace(obs) == removespace(expected)

    assert world.agents[0].holding is None
    assert world.agents[1].holding is None
    assert world.blender0.content.name == 'porkMeatcake'
    assert success == False
    obs, success, info = world.step(['goto_agent0_storage0', 'noop_agent1'])
    assert world.agents[0].location == world.storage0
    assert world.agents[1].location == world.blender0
    assert success == False
    assert all(x == y for x, y in zip(info['action_success'], [True, True]))

    obs, success, info = world.step(['noop_agent0', 'noop_agent1'])
    assert world.agents[0].location == world.storage0
    assert world.agents[1].location == world.blender0
    assert success == False
    assert all(x == y for x, y in zip(info['action_success'], [True, True]))

    obs, success, info = world.step(['noop_agent0', 'get_agent1_porkMeatcake_blender0'])
    assert world.agents[0].location == world.storage0
    assert world.agents[1].location == world.blender0
    assert world.agents[1].holding.name == 'porkMeatcake'
    assert world.blender0.content is None
    assert success == False
    assert all(x == y for x, y in zip(info['action_success'], [True, True]))

    obs, success, info = world.step(['noop_agent0', 'goto_agent1_servingtable0'])
    assert world.agents[0].location == world.storage0
    assert world.agents[1].location == world.servingtable0
    assert world.agents[1].holding.name == 'porkMeatcake'
    assert world.blender0.content is None
    assert success == False
    obs, success, info = world.step(['noop_agent0', 'put_agent1_porkMeatcake_servingtable0'])
    assert world.agents[0].location == world.storage0
    assert world.agents[1].location == world.servingtable0
    assert world.agents[1].holding is None
    assert world.blender0.content is None
    # when the wrong dish is given, the serving table will remained unchanged
    assert world.servingtable0.content.name == 'porkMeatcake'
    assert world.task_manager.accomplished_tasks() == []
    assert success == False
    assert obs.game_over == False

def test_2():
    world = World(2, recipe_filename='assets/recipe.json', task_filename='test/assets/tasks.json')
    world.reset(task_name='tomatoPasta')
    obs, success, info = world.step(['goto_agent0_storage0', 'goto_agent1_storage0'])

    assert world.agents[0].location == world.storage0
    assert world.agents[1].location == world.storage0
    assert world.agents[0].holding is None
    assert world.agents[1].holding is None
    world.step(['get_agent0_tomato_storage0', 'get_agent1_pasta_storage0'])
    assert world.agents[0].holding.name == 'tomato'
    assert world.agents[1].holding.name == 'pasta'
    world.step(['goto_agent0_pan0', 'goto_agent1_pot0'])
    assert world.agents[0].location == world.pan0
    assert world.agents[1].location == world.pot0

    # test error handling for invalid location
    obs, success, info = world.step(['noop_agent0', 'put_agent1_pasta_pasta0'])

    obs, success, info = world.step(['put_agent0_tomato_pan0', 'put_agent1_pasta_pot0'])

    assert world.agents[0].holding is None
    assert world.agents[1].holding is None
    assert world.pan0.content.name == 'tomato'
    assert world.pot0.content.name == 'pasta'
    obs, success, info = world.step(['activate_agent0_pan0', 'activate_agent1_pot0'])

    # assert removespace(obs) == removespace(expected)

    assert world.agents[0].holding is None
    assert world.agents[1].holding is None
    assert world.pot0.content.name == 'cookedPasta'
    assert world.pan0.content.name == 'sauteedTomato'
    assert success == False

    # test canot pickup from pan and pot
    obs, success, info = world.step(['get_agent0_sauteedTomato_pan0', 'get_agent1_cookedPasta_pot0'])
    assert world.pot0.content.name == 'cookedPasta'
    assert world.pan0.content.name == 'sauteedTomato'
    assert success == False
    assert all(x == y for x, y in zip(info['action_success'], [False, False]))

    # test no need to watch pan
    obs, success, info = world.step(['goto_agent0_storage0', 'goto_agent1_storage0'])
    assert world.pot0.content.name == 'cookedPasta'
    assert world.pan0.content.name == 'sauteedTomato'
    assert success == False
    assert all(x == y for x, y in zip(info['action_success'], [True, True]))

    obs, success, info = world.step(['goto_agent0_pan0', 'goto_agent1_pot0'])
    assert world.agents[0].location == world.pan0
    assert world.agents[1].location == world.pot0
    assert success == False
    assert all(x == y for x, y in zip(info['action_success'], [True, True]))

    obs, success, info = world.step(['get_agent0_sauteedTomato_pan0', 'get_agent1_cookedPasta_pot0'])
    assert world.agents[0].location == world.pan0
    assert world.agents[1].location == world.pot0
    assert world.agents[0].holding.name == 'sauteedTomato'
    assert world.agents[1].holding.name == 'cookedPasta'
    assert success == False
    assert all(x == y for x, y in zip(info['action_success'], [True, True]))

    obs, success, info = world.step(['put_agent0_sauteedTomato_pan0', 'goto_agent1_pan0'])
    assert world.agents[0].location == world.pan0
    assert world.agents[1].location == world.pan0
    assert world.agents[0].holding is None
    assert world.agents[1].holding.name == 'cookedPasta'
    assert success == False
    assert all(x == y for x, y in zip(info['action_success'], [True, True]))

    #TODO:
    #maybe need to discuss the order?
    obs, success, info = world.step(['noop_agent0', 'put_agent1_cookedPasta_pan0'])
    assert world.agents[0].location == world.pan0
    assert world.agents[1].location == world.pan0
    assert world.agents[0].holding is None
    assert world.agents[1].holding is None
    assert world.pan0.content.name == "sauteedTomato&cookedPasta"
    assert success == False
    assert all(x == y for x, y in zip(info['action_success'], [True, True]))

    obs, success, info = world.step(['activate_agent0_pan0', 'noop_agent1'])
    assert world.pan0.content.name == "tomatoPasta"

    obs, success, info = world.step(['noop_agent0', 'noop_agent1'])
    obs, success, info = world.step(['noop_agent0', 'noop_agent1'])
    obs, success, info = world.step(['noop_agent0', 'noop_agent1'])

    obs, success, info = world.step(['get_agent1_tomatoPasta_pan0', 'noop_agent0'])
    assert world.agents[1].holding.name == "tomatoPasta"

    obs, success, info = world.step(['noop_agent0', 'goto_agent1_servingtable0'])
    assert world.agents[0].location == world.pan0
    assert world.agents[1].location == world.servingtable0
    assert world.agents[1].holding.name == 'tomatoPasta'
    assert world.pan0.content is None
    assert world.pot0.content is None
    assert success == False
    obs, success, info = world.step(['noop_agent0', 'put_agent1_tomatoPasta_servingtable0'])
    assert world.agents[1].location == world.servingtable0
    assert world.agents[1].holding is None
    # when the right dish is given, the serving table will be cleared
    assert world.servingtable0.content is None
    assert world.task_manager.accomplished_tasks() == ['tomatoPasta']


# test game over when a task reach its lifetime
def test_3():
    world = World(2, task_lifetime=2, task_filename='test/assets/tasks.json')
    world.reset(task_name='tomatoPasta')
    obs, success, info = world.step(['noop_agent0', 'noop_agent1'])
    assert obs.game_over == False
    assert obs.current_tasks_name == ['tomatoPasta']
    assert obs.current_tasks_lifetime == [1]
    obs, success, info = world.step(['noop_agent0', 'noop_agent1'])
    assert obs.game_over == False
    assert obs.current_tasks_name == ['tomatoPasta']
    assert obs.current_tasks_lifetime == [0]
    obs, success, info = world.step(['noop_agent0', 'noop_agent1'])
    assert obs.game_over == False
    assert obs.just_failed == True
    assert obs.current_tasks_name == []

# test game over when the total steps is reach
def test_4():
    world = World(2, max_steps=2, task_filename='test/assets/tasks.json')
    world.reset(task_name='tomatoPasta')
    obs, success, info = world.step(['noop_agent0', 'noop_agent1'])
    assert obs.game_over == False
    assert obs.current_tasks_name == ['tomatoPasta']
    obs, success, info = world.step(['noop_agent0', 'noop_agent1'])
    assert obs.game_over == False
    assert obs.current_tasks_name == ['tomatoPasta']
    obs, success, info = world.step(['noop_agent0', 'noop_agent1'])
    assert obs.game_over == True
    assert obs.current_tasks_name == ['tomatoPasta']

# test new task being added, lifetime being decreased, and task being sorted by lifetime
def test_5():
    world = World(2, task_interval=2, task_lifetime=10, task_filename='test/assets/tasks.json')
    obs = world.reset(task_name='tomatoPasta')
    assert len(world.task_manager.current_tasks()) == 1
    assert obs.current_tasks_name[0] == 'tomatoPasta'
    assert obs.current_tasks_lifetime == [10]
    obs, success, info = world.step(['noop_agent0', 'noop_agent1'])
    assert len(world.task_manager.current_tasks()) == 1
    assert obs.current_tasks_name[0] == 'tomatoPasta'
    assert obs.current_tasks_lifetime == [9]
    obs, success, info = world.step(['noop_agent0', 'noop_agent1'])
    assert len(world.task_manager.current_tasks()) == 2
    assert obs.current_tasks_name[0] == 'tomatoPasta'
    assert obs.current_tasks_lifetime == [8, 10]
    obs, success, info = world.step(['noop_agent0', 'noop_agent1'])
    assert len(world.task_manager.current_tasks()) == 2
    assert obs.current_tasks_name[0] == 'tomatoPasta'
    assert obs.current_tasks_lifetime == [7, 9]
    obs, success, info = world.step(['noop_agent0', 'noop_agent1'])
    assert len(world.task_manager.current_tasks()) == 3
    assert obs.current_tasks_name[0] == 'tomatoPasta'
    assert obs.current_tasks_lifetime == [6, 8, 10]

# test multiple items for a single task
def test_6():
    world = World(2, recipe_filename='assets/recipe.json', task_filename='test/assets/tasks.json')
    obs = world.reset(task_name='two_porkMeatcake')
    assert len(world.task_manager.current_tasks()) == 1
    assert obs.current_tasks_name[0] == 'two_porkMeatcake'
    obs, success, info = world.step(['goto_agent0_storage0', 'goto_agent1_storage0'])
    obs, success, info = world.step(['get_agent0_flour_storage0', 'get_agent1_pork_storage0'])
    obs, success, info = world.step(['goto_agent0_blender0', 'goto_agent1_blender0'])
    obs, success, info = world.step(['put_agent0_flour_blender0', 'put_agent1_pork_blender0'])
    obs, success, info = world.step(['goto_agent0_storage0', 'activate_agent1_blender0'])
    obs, success, info = world.step(['get_agent0_flour_storage0', 'noop_agent1'])
    obs, success, info = world.step(['goto_agent0_blender0', 'get_agent1_porkMeatcake_blender0'])
    obs, success, info = world.step(['put_agent0_flour_blender0', 'goto_agent1_servingtable0'])
    obs, success, info = world.step(['goto_agent0_storage0', 'put_agent1_porkMeatcake_servingtable0'])
    # 1 porkMeatcake is finished
    assert success == False
    assert world.task_manager.accomplished_tasks() == []
    obs, success, info = world.step(['get_agent0_pork_storage0', 'goto_agent1_blender0'])
    obs, success, info = world.step(['goto_agent0_blender0', 'noop_agent1'])
    obs, success, info = world.step(['put_agent0_pork_blender0', 'activate_agent1_blender0'])
    obs, success, info = world.step(['noop_agent0', 'noop_agent1'])
    obs, success, info = world.step(['noop_agent0', 'get_agent1_porkMeatcake_blender0'])
    obs, success, info = world.step(['noop_agent0', 'goto_agent1_servingtable0'])
    obs, success, info = world.step(['noop_agent0', 'put_agent1_porkMeatcake_servingtable0'])
    assert obs.task_just_success == ['two_porkMeatcake']
    assert obs.task_just_success_location == ['servingtable0']
    assert success == True
    assert world.task_manager.accomplished_tasks() == ['two_porkMeatcake']

# test one goal is finished and the other is not
def test_7():
    world = World(2, task_interval=8, task_lifetime=10, task_filename='test/assets/tasks_single.json')
    obs = world.reset(task_name='porkMeatcake')
    assert len(world.task_manager.current_tasks()) == 1
    assert obs.current_tasks_name == ['porkMeatcake']
    assert obs.current_tasks_lifetime == [10]
    obs, success, info = world.step(['goto_agent0_storage0', 'goto_agent1_storage0'])
    obs, success, info = world.step(['get_agent0_flour_storage0', 'get_agent1_pork_storage0'])
    obs, success, info = world.step(['goto_agent0_blender0', 'goto_agent1_blender0'])
    obs, success, info = world.step(['put_agent0_flour_blender0', 'put_agent1_pork_blender0'])
    obs, success, info = world.step(['goto_agent0_storage0', 'activate_agent1_blender0'])
    obs, success, info = world.step(['get_agent0_flour_storage0', 'noop_agent1'])
    obs, success, info = world.step(['goto_agent0_blender0', 'get_agent1_porkMeatcake_blender0'])
    obs, success, info = world.step(['put_agent0_flour_blender0', 'goto_agent1_servingtable0'])
    assert len(world.task_manager.current_tasks()) == 2
    assert obs.current_tasks_name == ['porkMeatcake', 'porkMeatcake']
    assert obs.current_tasks_lifetime == [2, 10]
    obs, success, info = world.step(['goto_agent0_storage0', 'put_agent1_porkMeatcake_servingtable0'])
    assert obs.task_just_success == ['porkMeatcake']
    assert obs.task_just_success_location == ['servingtable0']
    assert success == True
    assert len(world.task_manager.current_tasks()) == 1
    assert obs.current_tasks_name == ['porkMeatcake']
    assert obs.current_tasks_lifetime == [9]
    assert world.task_manager.accomplished_tasks() == ['porkMeatcake']

# test two tasks success at the same time
def test_8():
    world = World(2, task_interval=10, task_lifetime=20, task_filename='test/assets/tasks_single.json')
    obs = world.reset() # with two serving tables and two blenders
    assert len(world.task_manager.current_tasks()) == 1
    assert obs.current_tasks_name[0] == 'porkMeatcake'
    assert obs.current_tasks_lifetime == [20]
    obs, success, info = world.step(['goto_agent0_storage0', 'goto_agent1_storage0'])
    obs, success, info = world.step(['get_agent0_flour_storage0', 'get_agent1_flour_storage0'])
    obs, success, info = world.step(['goto_agent0_blender0', 'goto_agent1_blender1'])
    obs, success, info = world.step(['put_agent0_flour_blender0', 'put_agent1_flour_blender1'])
    obs, success, info = world.step(['goto_agent0_storage0', 'goto_agent1_storage0'])
    obs, success, info = world.step(['get_agent0_pork_storage0', 'get_agent1_pork_storage0'])
    obs, success, info = world.step(['goto_agent0_blender0', 'goto_agent1_blender1'])
    obs, success, info = world.step(['put_agent0_pork_blender0', 'put_agent1_pork_blender1'])
    obs, success, info = world.step(['activate_agent0_blender0', 'activate_agent1_blender1'])
    obs, success, info = world.step(['noop_agent0', 'noop_agent1'])
    assert len(world.task_manager.current_tasks()) == 2
    assert obs.current_tasks_name == ['porkMeatcake', 'porkMeatcake']
    assert obs.current_tasks_lifetime == [10, 20]
    obs, success, info = world.step(['get_agent0_porkMeatcake_blender0', 'get_agent1_porkMeatcake_blender1'])
    obs, success, info = world.step(['goto_agent0_servingtable0', 'goto_agent1_servingtable1'])
    obs, success, info = world.step(['put_agent0_porkMeatcake_servingtable0', 'put_agent1_porkMeatcake_servingtable1'])
    assert obs.task_just_success == ['porkMeatcake', 'porkMeatcake']
    assert obs.task_just_success_location == ['servingtable0', 'servingtable1']
    assert success == True
    assert world.task_manager.accomplished_tasks() == ['porkMeatcake', 'porkMeatcake']

# test tool capacity
def test_10():
    world = World(2, task_filename='test/assets/tasks_single.json')
    obs = world.reset(task_name='porkMeatcake')
    obs, success, info = world.step(['goto_agent0_storage0', 'goto_agent1_storage0'])
    obs, success, info = world.step(['get_agent0_tomato_storage0', 'get_agent1_pork_storage0'])
    obs, success, info = world.step(['goto_agent0_chopboard0', 'goto_agent1_chopboard0'])
    obs, success, info = world.step(['put_agent0_tomato_chopboard0', 'put_agent1_pork_chopboard0'])
    assert all(x == y for x, y in zip(info['action_success'], [True, False]))
    obs, success, info = world.step(['goto_agent0_storage0', 'goto_agent1_storage0'])
    obs, success, info = world.step(['get_agent0_tomato_storage0', 'get_agent1_pork_storage0'])
    obs, success, info = world.step(['goto_agent0_blender0', 'goto_agent1_blender0'])
    obs, success, info = world.step(['put_agent0_tomato_blender0', 'put_agent1_pork_blender0'])
    assert all(x == y for x, y in zip(info['action_success'], [True, True]))
    obs, success, info = world.step(['goto_agent0_storage0', 'goto_agent1_storage0'])
    obs, success, info = world.step(['get_agent0_beef_storage0', 'get_agent1_flour_storage0'])
    obs, success, info = world.step(['goto_agent0_blender0', 'goto_agent1_blender0'])
    obs, success, info = world.step(['put_agent0_beef_blender0', 'put_agent1_flour_blender0'])
    assert all(x == y for x, y in zip(info['action_success'], [True, True]))
    obs, success, info = world.step(['goto_agent0_storage0', 'goto_agent1_storage0'])
    obs, success, info = world.step(['get_agent0_potato_storage0', 'get_agent1_onion_storage0'])
    obs, success, info = world.step(['goto_agent0_blender0', 'goto_agent1_blender0'])
    obs, success, info = world.step(['put_agent0_potato_blender0', 'put_agent1_onion_blender0'])
    assert all(x == y for x, y in zip(info['action_success'], [True, False]))

# test the limit of current tasks
def test_11():
    world = World(2, task_filename='test/assets/tasks_single.json', task_interval=3, task_lifetime=16, use_task_lifetime_interval_oracle=False, max_num_tasks=2)
    obs = world.reset(task_name='porkMeatcake')
    assert len(world.task_manager.current_tasks()) == 1
    obs, success, info = world.step(['noop_agent0', 'noop_agent1'])
    assert len(world.task_manager.current_tasks()) == 1
    obs, success, info = world.step(['noop_agent0', 'noop_agent1'])
    assert len(world.task_manager.current_tasks()) == 1
    obs, success, info = world.step(['noop_agent0', 'noop_agent1'])
    assert len(world.task_manager.current_tasks()) == 2
    obs, success, info = world.step(['noop_agent0', 'noop_agent1'])
    assert len(world.task_manager.current_tasks()) == 2
    obs, success, info = world.step(['noop_agent0', 'noop_agent1'])
    assert len(world.task_manager.current_tasks()) == 2
    obs, success, info = world.step(['noop_agent0', 'noop_agent1'])
    assert len(world.task_manager.current_tasks()) == 2
    obs, success, info = world.step(['noop_agent0', 'noop_agent1'])
    assert len(world.task_manager.current_tasks()) == 2
    obs, success, info = world.step(['goto_agent0_storage0', 'goto_agent1_storage0'])
    obs, success, info = world.step(['get_agent0_flour_storage0', 'get_agent1_pork_storage0'])
    obs, success, info = world.step(['goto_agent0_blender0', 'goto_agent1_blender0'])
    obs, success, info = world.step(['put_agent0_flour_blender0', 'put_agent1_pork_blender0'])
    obs, success, info = world.step(['goto_agent0_storage0', 'activate_agent1_blender0'])
    obs, success, info = world.step(['get_agent0_flour_storage0', 'noop_agent1'])
    obs, success, info = world.step(['goto_agent0_blender0', 'get_agent1_porkMeatcake_blender0'])
    obs, success, info = world.step(['put_agent0_flour_blender0', 'goto_agent1_servingtable0'])
    assert len(world.task_manager.current_tasks()) == 2
    obs, success, info = world.step(['goto_agent0_storage0', 'put_agent1_porkMeatcake_servingtable0'])
    assert len(world.task_manager.current_tasks()) == 1
    obs, success, info = world.step(['noop_agent0', 'noop_agent1'])
    assert len(world.task_manager.current_tasks()) == 1
    obs, success, info = world.step(['noop_agent0', 'noop_agent1'])
    assert len(world.task_manager.current_tasks()) == 2
    obs, success, info = world.step(['noop_agent0', 'noop_agent1'])
    obs, success, info = world.step(['noop_agent0', 'noop_agent1'])
    assert obs.game_over == False
    assert obs.just_failed == True
    assert len(world.task_manager.current_tasks()) == 1

if __name__ == '__main__':
    test_1()
    print('test_1 passed')