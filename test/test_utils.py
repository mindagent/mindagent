from levels.utils import compute_dependency, convert_to_prompt, removespace
from overcooked import World


def test_dependency():
    res, tools = compute_dependency('noexistscookie')
    assert res == []
    assert tools == []
    res, tools = compute_dependency('porkMeatcake')
    assert res == ['flour', 'pork', 'porkMeatcake'] or res == ['pork', 'flour', 'porkMeatcake']
    assert sorted(tools) == sorted(['blender', 'storage', 'servingtable'])

    res, tools = compute_dependency('tomatoPasta')
    assert res == ['tomato', 'sauteedTomato', 'pasta', 'cookedPasta', 'tomatoPasta'] or res == ['pasta', 'cookedPasta', 'tomato', 'sauteedTomato', 'tomatoPasta']
    assert sorted(tools) == sorted(['pot', 'pan', 'storage', 'servingtable'])

    res, tools, reward_mappings, mappings = compute_dependency('tomatoPasta', return_mapping=True)
    print(reward_mappings)
    print(mappings)

def test_convert_prompt():
    world = World(2, recipe_filename='assets/recipe.json', task_filename='test/assets/tasks.json')
    world.reset(task_name='tomatoPasta')
    obs, success, info = world.step(['goto_agent0_storage0', 'goto_agent1_storage0'])
    obs, success, info = world.step(['get_agent0_tomato_storage0', 'get_agent1_pasta_storage0'])

    # test no need to watch pan
    obs, success, info = world.step(['goto_agent0_pan0', 'goto_agent1_pot0'])
    obs, success, info = world.step(['put_agent0_tomato_pan0', 'put_agent1_pasta_pot0'])
    obs, success, info = world.step(['activate_agent0_pan0', 'activate_agent1_pot0'])
    obs, success, info = world.step(['noop_agent0', 'noop_agent1'])
    obs, success, info = world.step(['noop_agent0', 'noop_agent1'])
    obs, success, info = world.step(['noop_agent0', 'noop_agent1'])


    obs, success, info = world.step(['get_agent0_sauteedTomato_pan0', 'get_agent1_cookedPasta_pot0'])
    obs, success, info = world.step(['put_agent0_sauteedTomato_pan0', 'goto_agent1_pan0'])

    obs, success, info = world.step([ 'put_agent1_cookedPasta_pan0', 'activate_agent0_pan0'])

    obs, success, info = world.step(['noop_agent0', 'noop_agent1'])
    obs, success, info = world.step(['noop_agent0', 'noop_agent1'])
    obs, success, info = world.step(['noop_agent0', 'noop_agent1'])
    obs, success, info = world.step(['get_agent1_tomatoPasta_pan0', 'noop_agent0'])
    obs, success, info = world.step(['noop_agent0', 'goto_agent1_servingtable0'])

    prompt = convert_to_prompt(obs)
    expected = """at(agent0, pan0)
        hold(agent0, None)
        at(agent1, servingtable0)
        hold(agent1, tomatoPasta)
        inside(storage0, None)
        inside(servingtable0, None)
        inside(servingtable1, None)
        inside(blender0, None)
        inside(blender1, None)
        inside(chopboard0, None)
        inside(pot0, None)
        inside(pan0, None)
        inside(fryer0, None)
        inside(mixer0, None)"""
    # assert removespace(prompt) == removespace(expected)


    obs, success, info = world.step(['noop_agent0', 'put_agent1_tomatoPasta_servingtable0'])
    assert world.task_manager.accomplished_tasks() == ['tomatoPasta']
