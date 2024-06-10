import datetime
import glob
import json
import os
import pickle
import re
import shutil
import sys
from typing import Dict, List

import git
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from uuid import uuid1

from flask import Flask, jsonify, render_template, request

from levels.utils import (StepReturnType, compute_dependency,
                          convert_to_prompt, draw_dependecy_graph,
                          extract_agent_id)
from overcooked import World
from utils.llm import chat_llm, prepend_history, rules

app = Flask(__name__)
app.jinja_env.filters['zip'] = zip

world_dict : Dict[str, World] = {}
history_dict: Dict[str, List] = {}

feedback = '-execution error messages:\n  --  []\n'
suggestions = '-execution suggestions:\n  --  []\n'

@app.route('/')
def start():
    with open("assets/tasks_level_final.json", 'r') as f:
        data = json.load(f)

    levels = []
    for key, value in data.items():
        levels.append({"value": key, 'label': key})


    return render_template('start.html', levels=levels)

@app.route('/game' ,methods = ['POST', 'GET'])
def game_handle():
    global world_dict, feedback, suggestions

    if request.method == 'POST':
        result = request.form
        json_result = dict(result)
        user_id = json_result['user_id']

        world = world_dict[user_id]
        num_agents = world.num_agents
        actions = []

        for index in range(num_agents):
            if f'agent{index}_id' not in json_result or f'agent{index}_action' not in json_result:
                continue

            agent_id = int(json_result[f"agent{index}_id"])

            action = json_result[f'agent{index}_action']
            action = action.split(' ')
            action_type = action[0]

            if action_type == 'noop':
                str_action = f'noop_agent{agent_id}'
            elif action_type == 'goto':
                str_action = f'goto_agent{agent_id}_{action[1]}'
            elif action_type == 'get':
                if len(action) == 3:
                    str_action = f'get_agent{agent_id}_{action[1]}_{action[2]}'
                elif len(action) == 2:
                    str_action = f'get_agent{agent_id}_{action[1]}'
                else:
                    raise RuntimeError("get arguments wrong")

            elif action_type == 'put':
                if len(action) == 3:
                    str_action = f'put_agent{agent_id}_{action[1]}_{action[2]}'
                elif len(action) == 2:
                    str_action = f'put_agent{agent_id}_{action[1]}'
                else:
                    raise RuntimeError("put arguments wrong")

            elif action_type == 'activate':
                str_action = f'activate_agent{agent_id}_{action[1]}'

            actions.append(str_action)

        def prepare_history(num_agents):
           
            if num_agents == 2:
                asset_file = './assets/amt_examples_human/prompt_2agent_level1.txt'
            elif num_agents == 3:
                asset_file = './assets/amt_examples_human/prompt_3agent_level2.txt'
            elif num_agents == 4:
                asset_file = './assets/amt_examples_human/prompt_4agent_level2.txt'
            else:
                raise Exception(f"{num_agents} not supported")

            example = open(asset_file, 'r').read().split('###\n')[1].split('***\n')
            example_history = []
            for idx, exp in enumerate(example):
                if idx % 2 == 0:
                    example_history.append(("user", exp))
                else:
                    example_history.append(("assistant", exp))
            pre_prompt = ("user" , rules(world, True))
            # info_prompt = ("user", f"There are {world.num_agents} agents available, so you can execute {world.num_agents} actions at a time.\n")
            info_prompt = ("user", f"In this game, there is one human, denote as agent0, you should read the human's action and control the other {world.num_agents-1} agents to collaborate with the human. Therefore, you should plan {world.num_agents-1} actions at a time.")
            history = [pre_prompt] + example_history + [info_prompt]
            return history

        def gpt_agent():
            initial_history_length = len(prepare_history(world.num_agents))
            if user_id not in  history_dict:
                history_dict[user_id] = prepare_history(world.num_agents)

            look_ahead_steps = 5

            obs = world.all_state()
            prompt =  feedback +  suggestions + convert_to_prompt(obs) + '-action:\n'
            for action in actions:
                prompt += action + '\n'

            if len(history_dict[user_id]) < look_ahead_steps + initial_history_length:
                history_dict[user_id] = prepend_history(history_dict[user_id], prompt, role="user", verbose=False)
            else:
                history_dict[user_id] = history_dict[user_id][:initial_history_length] + history_dict[user_id][initial_history_length+2:]
                history_dict[user_id] = prepend_history(history_dict[user_id], prompt, role="user", verbose=False)

            plan = chat_llm(history_dict[user_id], temperature=0.2, model="gpt-4")
            # print('=======')
            # print(history_dict[user_id][-1][1])
            # print('-------')
            # print(plan)
            # print('=======')
            plan = plan.split('\n')[:world.num_agents-1]
            plan = actions + plan
            return plan
        def random_agent():
            
            actionss = world.available_actions()[1:]
            
            results = []
            for idx, action_iter in enumerate(actionss):
                plan_action = random.choice(action_iter)
                components = plan_action.split(' ')
                verb  = components[0]
                if len(components) == 2:
                    args =  [f'agent{idx+1}', components[-1]]
                elif len(components) == 3:
                    location = components[-1]
                    noun = components[1]
                    args =  [f'agent{idx+1}', noun ,location]
                else:
                    args = [f'agent{idx+1}']

                plan_action_final = '_'.join([verb] + args) 
                results.append(plan_action_final)

            return actions + results


        if world.play_mode == 'play_with_human' and world.num_agents > 1:
            actions = gpt_agent()
        elif world.play_mode == 'random_agent':
            actions = random_agent()


        world.step(actions)
        if world.play_mode == 'play_with_human' and world.num_agents > 1:
            feedback = '-execution error messages:\n  --  ' + str(world.feedback) + '\n'
            suggestions = '-execution suggestions:\n  --  ' + str(world.suggestions) + '\n'
            to_add = ''
            # skip human's action, which is not planned by LLM
            for i in range(1, world.num_agents):
                if i < len(actions):
                    to_add += actions[i] + '\n'

            # history = prepend_history(history, plan[0] + '\n' + plan[1])
            history_dict[user_id] = prepend_history(history_dict[user_id], to_add, role='assistant')

        state = get_state(user_id)
        agents, tools = state_to_struct(state)

        available_actions = world.available_actions()
        num_agents = world.num_agents
        num_agents = [int(i) for i in range(num_agents)]
        if world.play_mode in ['random_agent', 'play_with_human']:
            num_agents = num_agents[:1]
            available_actions = available_actions[:1]

        tasks_name  = state.current_tasks_name
        tasks_lifetime = state.current_tasks_lifetime

        state = world.done()
        game_over = state.game_over

        accomplished_tasks = ', '.join(world.task_manager.accomplished_tasks())
        previous_actions = world.previous_actions
        if not game_over:
            return render_template("text_game.html", tasks_name = tasks_name, tasks_lifetime = tasks_lifetime,
                                agents=agents, tools=tools, available_actions = available_actions,
                                num_agents=num_agents, user_id = user_id,
                                max_step = world.max_steps, time_step = world.time_step,
                                accomplished_orders = accomplished_tasks, level_name=world.level,
                                previous_actions=previous_actions)
        else:

            file_name = f"{world.level}_ {world.num_agents}_{str(user_id)}"
            repo = git.Repo(search_parent_directories=True)
            sha = repo.head.object.hexsha
            output_content = {
                'level_name': world.level,
                'num_agents': world.num_agents,
                'user_id': user_id,
                'recipe_filename': world.recipe_filename,
                'action_history': world.action_history,
                'state_history': world.state_history,
                'commit_id': sha,
                'world_max_steps': world.max_steps,
                'world_seed': world.seed,
                'world_alpha': world.alpha,
                'world_beta': world.beta,
                'world_use_task_lifetime_interval_oracle': world.use_task_lifetime_interval_oracle,
                'world_play_mode': world.play_mode,
            }

            if not os.path.exists('logs'):
                os.makedirs('logs')

            with open(f'logs/{file_name}.pkl', 'wb') as f:
                pickle.dump(output_content, f)

            del world_dict[user_id]
            if user_id in history_dict:
                del history_dict[user_id]
            return render_template('end.html', user_id=user_id, num_agents = world.num_agents)


@app.route('/result',methods = ['POST', 'GET'])
def result():
    global world_dict


    if request.method == 'POST':
        current_datetime = datetime.datetime.now()
        current_datetime_in_seconds = current_datetime.timestamp()
        seed = int(current_datetime_in_seconds)
        result = request.form
        json_result = dict(result)
        user_id = str(uuid1())
        current_level = result['level']
        game_mode = result['game_mode']
        if game_mode == 'collaboration':
            play_mode = 'play_with_human'
        elif game_mode == 'random':
            play_mode = 'random_agent'
        else:
            play_mode = 'gpt_agent'

        # if result['numAgents'] == '1':
        #     alpha = 3
        # elif result['numAgents'] == '2':
        #     alpha = 2.5
        # else:
        #     alpha = 2.0
        alpha = 2.0
        world = World(override_agent=True, max_steps=60, alpha=alpha,
                        seed=seed, num_agents=int(result['numAgents']), 
                      recipe_filename="assets/recipe.json",
                      task_filename="assets/tasks_level_final.json", level=current_level,
                    use_task_lifetime_interval_oracle=True,
                      play_mode = play_mode)

        world_dict[user_id] = world
        world.reset(task_name=None)

        state = get_state(user_id)

        agents, tools = state_to_struct(state)

        available_actions = world.available_actions()
        num_agents = world.num_agents
        num_agents = [int(i) for i in range(num_agents)]

        if world.play_mode in ['random_agent', 'play_with_human']:
            num_agents = num_agents[:1]
            available_actions = available_actions[:1]

        tasks_name  = state.current_tasks_name
        tasks_lifetime = state.current_tasks_lifetime

        accomplished_tasks = ', '.join(world.task_manager.accomplished_tasks())
        previous_actions = world.previous_actions
        print('previous actions: ', previous_actions)
        return render_template("text_game.html", tasks_name = tasks_name, tasks_lifetime = tasks_lifetime,
                                agents=agents, tools=tools, available_actions = available_actions,
                                num_agents=num_agents, user_id = user_id,
                                max_step = world.max_steps, time_step = world.time_step,
                                accomplished_orders = accomplished_tasks, level_name=world.level,
                                previous_actions=previous_actions)



def get_state(user_id):
    global world_dict
    return world_dict[user_id].all_state()


def state_to_struct(state: StepReturnType):
    agents = state.agents
    tools = state.locations

    for tool in tools:
        content = tool['content']
        if content is not None:
            tool['objects'] = content.split('_')
        else:
            tool['objects'] = None
        if 'storage' in tool['id']:
            tool['objects'] = ['everything']


    return agents, tools

@app.route('/get_sub_list', methods=['POST'])
def get_sub_list():

    agent_id = int(request.form['mainOption'])
    user_id = str(request.form['user_id'])
    world = world_dict[user_id]
    available_actions = world.available_actions()
    return_actions = available_actions[agent_id]
    return_actions = [
        {'value': option, 'text': option} for option in return_actions
    ]
    return jsonify(return_actions)

@app.route('/recipe', methods=['GET', 'POST'])
def recipe():
    if request.method == 'POST':
        level = request.form.get('level_name')
    else:
        level = 'level_0'

    with open('assets/tasks_level_final.json', 'r') as f:
        task_level = json.load(f)
    dishes = task_level[level]['task_list']

    dishes_name = list(dishes.keys())

    fl = glob.glob('webpage/static/images/recipe/*')
    if len(fl) > 500:
        for i in fl:
            if os.path.isfile(i):
                os.remove(i)
    recipes = []
    for dish in dishes_name:
        G = compute_dependency(dish, return_graph=True)
        path = draw_dependecy_graph(G)
        folder_name = 'webpage/static/images/recipe/'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        shutil.copyfile(path, os.path.join(folder_name, path.split('/')[-1]))
        img_path = os.path.join('images/recipe', path.split('/')[-1])
        recipes.append({
            'dish': dish,
            'image_path': img_path
        })

    return render_template('recipe.html', recipes=recipes)

@app.route('/guide', methods=['GET', 'POST'])
def guide():
    return render_template('guide.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
