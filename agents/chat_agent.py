import argparse
import json
import os
import sys
import time
import re
import openai
import tiktoken

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from levels.utils import convert_to_prompt
from overcooked import World
from utils.llm import chat_llm, chat_llm_vicuna, prepend_history, rules


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def extract_actions(text):
    # List of action types
    action_types = ["noop", "goto", "put", "activate", "get"]
    
    # Pattern for the actions
    pattern = r'((' + '|'.join(action_types) + r')_agent\d+(_[a-zA-Z0-9_]+)?)'
    
    matches = re.findall(pattern, text)
    
    # Extracting just the full action names from the returned tuples
    actions = [match[0] for match in matches]
    return actions



def main(model, with_feedback_arg, with_notes_arg, num_agents, level_to_run):
    NUM_AGENTS = num_agents
    alphas = [ 1.0, 1.5, 2.0, 2.5, 3.0 ]
    if level_to_run == 'all':
        levels = [  
                    'level_0',
                    'level_1',
                    'level_2',
                    'level_3',
                    'level_4',
                    'level_5',
                    'level_6',
                    'level_7',
                    'level_8',
                    'level_9',
                    'level_10',
                    'level_11', 
                    'level_12']
    else:
        levels = [level_to_run]
    with_feedback = with_feedback_arg
    with_notes = with_notes_arg

    

    for level in levels:
        if with_feedback is False:
            save_file_name = f'result_{level}_{NUM_AGENTS}_wo_feedback_{model}.json'
        elif with_notes is False:
            save_file_name = f'result_{level}_{NUM_AGENTS}_wo_notes_{model}.json'
        else:
            save_file_name = f'result_{level}_{NUM_AGENTS}_{model}.json'
        if os.path.exists(save_file_name):
            with open(save_file_name, 'r') as f:
                table = json.load(f)
        else:
            table = {}
        
        for alpha in alphas :
            if str(alpha) in table.keys():
                continue
            env = World(recipe_filename='./assets/recipe.json', task_filename='./assets/tasks_level_iclr.json',
                         level=level, use_task_lifetime_interval_oracle=True,
                        alpha=alpha, beta=2.5, num_agents=NUM_AGENTS, override_agent=True)
            max_episode = 3
            max_steps = 60

            if model == 'vicuna':
                asset_file = './assets/prompt_2_short.txt'
            else:
                if num_agents == 1:
                    asset_file = './assets/amt_examples/prompt_1agent_level1.txt'
                elif num_agents == 2:
                    asset_file = './assets/amt_examples/prompt_2agent_level1.txt'
                elif num_agents == 3:
                    asset_file = './assets/amt_examples/prompt_3agent_level2.txt'
                elif num_agents == 4:
                    asset_file = './assets/amt_examples/prompt_4agent_level2.txt'

            example = open(asset_file, 'r').read().split('###\n')[1].split('***\n')
            example_history = []
            for idx, exp in enumerate(example):
                if idx % 2 == 0:
                    example_history.append(("user", exp))
                else:
                    example_history.append(("assistant", exp))

            num_success = 0
            total = 0
            success = 0
            failed = 0

            if model == 'vicuna':
                look_ahead_steps = 3
            else:
                look_ahead_steps = 5

            # [("user", text), ("user", text), ('assistant', text)]

            total_action_histories = []
            total_action_success_histories = []
            total_prompts = []
            context = ''
            for eps_id in range(max_episode):
                obs = env.reset()
                if model == 'vicuna':
                    pre_prompt = ("user" , rules(env, False))
                else:
                    pre_prompt = ("user" , rules(env, with_notes))
                info_prompt = ("user", f"In this game, There are {env.num_agents} agents available and no human, so you should control all the {env.num_agents} agents, and plan {env.num_agents} actions at a time.\n")

                if model == 'palm-2':
                    history = example_history + [info_prompt]
                else:
                    history = [pre_prompt] + example_history + [info_prompt]

                context = pre_prompt[1] + '\n' + info_prompt[1]

                goal = env.task
                step = 0
                done = False
                action_histories = []
                action_success_histories = []
                prompt_history = []
                initial_history_length = len(history)

                if with_feedback:
                    feedback = '-execution error messages:\n  --  []\n'
                    suggestions = '-execution suggestions:\n  --  []\n'
                else:
                    feedback = ''
                    suggestions = ''
                while True:
                    # update history
                    prompt =  feedback +  suggestions + convert_to_prompt(obs) + '-action:\n'
                    raw_prompt = convert_to_prompt(obs)
                    prompt_history.append(raw_prompt)

                    # cap message length
                    if len(history) < look_ahead_steps + initial_history_length:
                        history = prepend_history(history, prompt)
                    else:
                        history = history[:initial_history_length] + history[initial_history_length+2:]
                        history = prepend_history(history, prompt)


                    if step >= max_steps:
                        break
                    cnt = 0

                    # update history, plan and execute
                    if model == 'vicuna':
                        plan = chat_llm_vicuna(history=history, temperature=0.1)
                    else:
                        if model == 'palm-2':
                            time.sleep(3)
                            plan = chat_llm(history, temperature=0.1, model=model, context=context)
                        else:
                            print('alpha: ', alpha)
                            plan = chat_llm(history, temperature=0.1, model=model)
                    # print('plan:', plan)
                    # 
                    # if model == 'claude-2':
                    print('plan:', plan)
                    try:
                        plan = extract_actions(plan)
                    except:
                        plan = []
                    # else:
                    #     plan = plan.split('\n')[:env.num_agents]

                    if plan:
                        obs, done, info = env.step(plan)
                    action_histories.append(plan)
                    if with_feedback:
                        feedback = '-execution error messages:\n  --  ' + str(env.feedback) + '\n'
                        suggestions = '-execution suggestions:\n  --  ' + str(env.suggestions) + '\n'


                    to_add = ''
                    for i in range(env.num_agents):
                        if i < len(plan):
                            to_add += plan[i] + '\n'

                    # history = prepend_history(history, plan[0] + '\n' + plan[1])
                    history = prepend_history(history, to_add, role='assistant')
                    print(info['action_success'])
                    action_success_histories.append(env.action_success_history)
                    step += 1

                total += env.success_count + env.failed_count + len(env.task_manager._current_task_list)
                success += env.success_count
                failed += env.failed_count
                total_action_histories.append(action_histories)
                total_action_success_histories.append(env.action_success_history)
                total_prompts.append(prompt_history)

            table[alpha] = {
                'total' : total,
                'success' : success,
                'failed': failed,
                'alpha': alpha,
                'noop_count': env.noop_count,
                'action_history': total_action_histories,
                'action_success_history': total_action_success_histories,
                'prompt_history': total_prompts
            }
            if with_feedback is False:
                with open(f'result_{level}_{NUM_AGENTS}_wo_feedback_{model}.json', 'w') as fp:
                    json.dump(table, fp)
            elif with_notes is False:
                with open(f'result_{level}_{NUM_AGENTS}_wo_notes_{model}.json', 'w') as fp:
                    json.dump(table, fp)
            else:

                with open(f'result_{level}_{NUM_AGENTS}_{model}.json', 'w') as fp:
                    json.dump(table, fp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="invoking GPT")

    # Add the arguments
    parser.add_argument('model_name', metavar='model_name', type=str, choices=['gpt-4', 'gpt-4-azure',  'gpt-3.5-turbo', 'vicuna', 'claude-2', 'palm-2'], help='model to use')
    parser.add_argument('--without_feedback', action='store_true', help='without feedback')
    parser.add_argument('--without_notes', action='store_true', help='without notes')
    parser.add_argument('--num_agents', metavar='num_agents', type=int, required=True ,help='number of agents')
    parser.add_argument('--level', metavar='level', type=str, required=True ,help='level of the game')
    # Parse the arguments
    args = parser.parse_args()

    total = int(args.without_feedback) + int(args.without_notes)
    assert total <= 1

    main(model = args.model_name, with_feedback_arg= not (args.without_feedback),
          with_notes_arg=not (args.without_notes), 
        num_agents = int(args.num_agents), level_to_run = args.level)