import json
import math
import re
import uuid

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('agg')
import networkx as nx

from levels.constants import (ALPHA, BETA, GAMMA, StepReturnType,
                              all_location_types, all_objecsts,
                              base_ingridients, occupied_time)


def filter_recipe(recipe, tasks):
    tmp = tasks
    new_recipe = {}
    while len(tmp):
        item = tmp.pop(0)
        new_recipe[item] = recipe[item]
        for i in recipe[item]['ingredients']:
            if i in recipe:
                tmp.append(i)
    return new_recipe


def removespace(a):
        return a.replace('\t', '').replace('\n', '').replace(' ', '')

def extract_agent_id(string):
    match = re.search(r'agent(\d+)', string)
    if match:
        return match.group(1)
    return None

def compute_wait_time(task_name):
    with open('assets/recipe.json', 'r') as f:
        recipes = json.load(f)

    if task_name not in recipes:
        return -1
    # Define the set of nodes and edges in the graph
    nodes = []
    edges = []
    queue  = [task_name]

    wait_time = 0
    while len(queue) > 0:
        front = queue.pop(0)
        nodes.append(front)
        if front in base_ingridients:
            continue
        children = recipes[front]['ingredients']
        # TODO(jxma): minus 1 to substract the "activate" action
        if occupied_time[recipes[front]['location']] > 1:
            wait_time += occupied_time[recipes[front]['location']] - 1

        for child in children:
            edges.append((child, front, {'label': recipes[front]['location']}))
        queue.extend(children)
    
    # Create the graph object
    G = nx.DiGraph()

    # Add the nodes and edges to the graph
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # Check if the graph is a DAG (i.e., has no cycles)
    if not nx.is_directed_acyclic_graph(G):
        print("Graph contains a cycle")

    # Get the topological sort order of the nodes (i.e., the order in which to process the dependencies)
    topological_order = list(nx.topological_sort(G))
    # TODO(jxma): we tentatively view 1 edge as 3 time steps (get, goto, put)
    total_wait_time = wait_time + len(topological_order) * 3
    return total_wait_time

def compute_dependency(task_name, return_mapping=False, return_graph=False):
    # return the following:
    # - topological order of the ingridients
    # - tools required to finsih the task

    with open('assets/recipe.json', 'r') as f:
        recipes = json.load(f)

    if task_name not in recipes:
        if return_mapping:
            return [], [], [], []
        else:
            return [], []
    # Define the set of nodes and edges in the graph
    nodes = []
    edges = []
    queue  = [task_name]
    stack = [task_name]
    tools = []
    mappings = {}

    depth_mapping = {task_name: 1 }
    while len(stack) > 0 :
        front = stack.pop(-1)
        if front in base_ingridients:
            continue
        children = recipes[front]['ingredients']
        for child in children:
            depth_mapping[child] = depth_mapping[front] + 1
        stack.extend(children)

    reward_mapping = {key: 10**(-value*1.0 + 2) for key, value in depth_mapping.items()}
    
    while len(queue) > 0:
        front = queue.pop(0)
        nodes.append(front)
        if front in base_ingridients:
            continue
        children = recipes[front]['ingredients']
        tools.append(recipes[front]['location'])
        if recipes[front]['location'] not in mappings:
            mappings[recipes[front]['location']] = [children]
        else:
            mappings[recipes[front]['location']].append(children)

        for child in children:
            edges.append((child, front, {'label': recipes[front]['location']}))
        queue.extend(children)

    tools.extend(['storage', 'servingtable'])
    tools = sorted(list(set(tools)))
    # Create the graph object
    G = nx.DiGraph()

    # Add the nodes and edges to the graph
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # Check if the graph is a DAG (i.e., has no cycles)
    if not nx.is_directed_acyclic_graph(G):
        print("Graph contains a cycle")

    # Get the topological sort order of the nodes (i.e., the order in which to process the dependencies)
    topological_order = list(nx.topological_sort(G))

    if return_graph:
        return G

    # Print the topological sort order
    if return_mapping:
        return topological_order, tools, reward_mapping, mappings
    else:
        return topological_order, tools

def draw_dependecy_graph(G, add_serving_table=True):
    plt.figure(figsize=(40, 13))
    if add_serving_table:
        G.add_node('success')
        final_node = list(nx.topological_sort(G))[-1]
        G.add_edge(final_node, 'success', label='serving table')
    # pos = nx.kamada_kawai_layout(G)  # positions for all nodes
    pos = nx.fruchterman_reingold_layout(G)
    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=1400)
    # edges
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=60, node_size=
    3000, width=6)
    # labels
    label_pos = {node: (x+0.07, y+0.07) for node, (x, y) in pos.items()}
    # Adjust label positions
    nx.draw_networkx_labels(G, label_pos, font_size=50, font_family='sans-serif')
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=50)
    plt.axis('off')
    path = f'/tmp/{str(uuid.uuid4())}.svg'
    plt.savefig(path, format='svg')
    return path


def compute_lifetime_task_intervals(task_name:str, num_agents:int, alpha=None, beta=None):
    topological_order, tools  = compute_dependency(task_name=task_name)
    total_wait_time = compute_wait_time(task_name=task_name)
    # task_interval = alpha * minimal completion time of a single agent (alpha ~= averaged steps per sub goal)
    if alpha is None:
        alpha = ALPHA
    if beta is None:
        beta = BETA

    # TODO(jxma): our hyperparameter is calibrated based on the depth of the task recipe DAG; 
    # we tentatively view 1 edge as 3 time steps (get, goto, put)
    task_interval = alpha * total_wait_time * 0.3
    # lifetime = beta * task_interval
    lifetime = beta * task_interval

    # ma_task_interval = gamma(num_agents) * task_interval
    task_interval = GAMMA(num_agents) * task_interval
    # ma_lifetime = gamma(num_agents) * task_interval
    lifetime = GAMMA(num_agents) * lifetime
    return math.ceil(lifetime), math.ceil(task_interval)

def get_goal_prompt(state: StepReturnType):
    prompt = ""
    prompt += "current dishes:\n"
    for i, j in zip(state.current_tasks_name, state.current_tasks_lifetime):
        prompt += f"    name: {i} lifetime: {j}\n"
    return prompt

def get_state_prompt(state: StepReturnType):
    num_agents = len(state.agents)

    prompt = ""
    prompt += f"current game step: {state.current_step}\n"
    prompt += f"maximum game steps: {state.max_steps}\n"

    prompt += '\n-agent state:\n'
    # agent state
    for i in range(num_agents):
        prompt += f"at(agent{i}, {state.agents[i]['location'] })\n"
        prompt += f"hold(agent{i}, {state.agents[i]['hold']})\n"
        if state.agents[i]['occupy']:
            prompt += f"occupy(agent{i}, {str(bool(state.agents[i]['occupy']))})\n"

    prompt += '\n-kitchen state:\n'
    # kitchen state
    for location in state.locations:
        content = location['content']
        prompt += f"inside({location['id']}, {content})\n"
        if location['occupy']:
            prompt += f"occupy({location['id']})\n"

    prompt += "\n-accomplished task:\n"
    for task in state.accomplished_tasks:
        prompt += f"{task}, "

    prompt += "\n\n"
    return prompt

def convert_to_prompt(state: StepReturnType):
    prompt = ""
    num_agents = len(state.agents)

    prompt += "-game state:\n"
    # game state
    prompt += f"current game level: {state.current_level}\n"
    prompt += "current dishes:\n"
    for i, j in zip(state.current_tasks_name, state.current_tasks_lifetime):
        prompt += f"    name: {i} lifetime: {j}\n"
    prompt += f"current game step: {state.current_step}\n"
    prompt += f"maximum game steps: {state.max_steps}\n"

    prompt += '\n-agent state:\n'
    # agent state
    for i in range(num_agents):
        prompt += f"at(agent{i}, {state.agents[i]['location'] })\n"
        prompt += f"hold(agent{i}, {state.agents[i]['hold']})\n"
        if state.agents[i]['occupy']:
            prompt += f"occupy(agent{i}, {str(bool(state.agents[i]['occupy']))})\n"

    prompt += '\n-kitchen state:\n'
    # kitchen state
    for location in state.locations:
        content = location['content']
        prompt += f"inside({location['id']}, {content})\n"
        if location['occupy']:
            prompt += f"occupy({location['id']})\n"

    prompt += "\n-accomplished task:\n"
    for task in state.accomplished_tasks:
        prompt += f"{task}, "

    prompt += "\n\n"

    return prompt

def convert_to_state(state: StepReturnType):
    import numpy as np

    # current tasks names [*5],  current lifetime[*5], $ 5
    # current game step, maximum game step,  $ 2
    # agent 1 location, agent 1 holding , agent1 occupied,  agent 2 location, agent 2 holding, agent 2 occupied $ 6
    # total_tools *4 and its contents.
    num_tools = len(all_location_types)
    num_objects = len(all_objecsts)

    feature = np.zeros(64)

    start_idx = 0
    for idx, task in enumerate(state.current_tasks_name):
        num = all_objecsts.index(task)
        feature[start_idx] = num
        start_idx += 1

    start_idx = 5
    for lifetime in state.current_tasks_lifetime:
        feature[start_idx] = lifetime
        start_idx += 1

    start_idx = 10
    feature[start_idx] = state.current_step
    start_idx = 11
    feature[start_idx] = state.max_steps

    start_idx = 12
    loc_id = int(state.agents[0]['location'][-1])
    loc = state.agents[0]['location'][:-1]
    feature[start_idx] = all_location_types.index(loc)*3 + loc_id

    start_idx = 13
    if state.agents[0]['hold']:
        if state.agents[0]['hold'] == 'waste':
            feature[start_idx] = -2
        else:
            feature[start_idx] = all_objecsts.index(state.agents[0]['hold'])
    else:
        feature[start_idx] = -1

    start_idx = 14
    feature[start_idx] = int(state.agents[0]['occupy'])

    start_idx = 15

    loc_id = int(state.agents[1]['location'][-1])
    loc = state.agents[1]['location'][:-1]
    feature[start_idx] = all_location_types.index(loc)*3 + loc_id


    start_idx = 16
    if state.agents[1]['hold']:
        if state.agents[1]['hold'] == 'waste':
            feature[start_idx] = -2
        else:
            feature[start_idx] = all_objecsts.index(state.agents[1]['hold'])
    else:
        feature[start_idx] = -1
    start_idx = 17
    feature[start_idx] = int(state.agents[1]['occupy'])

    start_idx = 18
    max_num = 18

    for location in state.locations:
        content = location['content']
        if content:

            content_idx = 0
            for c in content.split('&'):
                if c == 'waste':
                    content_idx = -2
                    break
                tmp = 1 << all_objecsts.index(c)
                content_idx += tmp

        else:
            content_idx = -1
        loc_id = int(location['id'][-1])
        loc = location['id'][:-1]

        loc_num = all_location_types.index(loc)*4 + loc_id + start_idx
        occupancy_index = all_location_types.index(loc)*4 + 3 + start_idx

        if location['occupy']:
            feature[occupancy_index] = 1

        feature[loc_num] = content_idx
        max_num = max(max_num, loc_num)
    return feature


def load_data():
    import os
    logs = os.listdir('logs')
    logs = ['logs/'+log for log in logs]
    import pickle
    all_data = []
    for log in logs:
        with open(log, 'rb') as f:
            data = pickle.load(f)
            all_data.append(data)
    print(all_data[0])
