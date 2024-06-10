import json
import os
import sys
import time

import openai
import tiktoken

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from levels.utils import convert_to_prompt
from overcooked import World
from utils.llm import chat_llm_vicuna, prepend_history, rules


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

NUM_AGENTS = 2
with_feedback = True


def main():
    alphas = [1.0, 1.5, 2.0, 2.5, 3.0]
    levels = [ 'level_2', 'level_3', 'level_4', 'level_5', 'level_6',  'level_7', 'level_8',
              'level_9', 'level_10',
              'level_11']

    for level in levels:
        table = {}
        for alpha in alphas :
            env = World(recipe_filename='./assets/recipe.json', level=level, use_task_lifetime_interval_oracle=True,
                        alpha=alpha, beta=2.5, num_agents=NUM_AGENTS, override_agent=True)
            max_episode = 3
            max_steps = 60

            example = open('./assets/prompt_2.txt', 'r').read().split('###\n')[1].split('***\n')
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

            look_ahead_steps = 5

            total_action_histories = []
            total_action_success_histories = []
            total_prompts = []
            for eps_id in range(max_episode):
                obs = env.reset()
                print(rules(env))
                pre_prompt = ("user" , rules(env))
                info_prompt = ("user", f"There are {env.num_agents} agents available, so you can execute {env.num_agents} actions at a time.\n")

                history = [pre_prompt] + example_history + [info_prompt]
                rules_text = rules(env)

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

                    # cap message length to 25
                    if len(history) < look_ahead_steps + initial_history_length:
                        history = prepend_history(history, prompt)
                    else:
                        history = history[:initial_history_length] + history[initial_history_length+2:]
                        history = prepend_history(history, prompt)


                    if step >= max_steps:
                        break
                    cnt = 0

                    # update history, plan and execute
                    plan = chat_llm_vicuna(history, temperature=0.2)
                    plan = plan.split('\n')[:env.num_agents]


                    obs, done, info = env.step(plan)
                    action_histories.append(plan)
                    if with_feedback:
                        feedback = '-execution error messages:\n  --  ' + str(env.feedback) + '\n'
                        suggestions = '-execution suggestions:\n  --  ' + str(env.suggestions) + '\n'

                    to_add = ''
                    for i in range(env.num_agents):
                        if i < len(plan):
                            to_add += plan[i] + '\n'

                    try:
                        # history = prepend_history(history, plan[0] + '\n' + plan[1])
                        history = prepend_history(history, to_add, role='assistant')
                    except:
                        history = prepend_history(history, '', role='assistant')
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
                with open(f'result_{level}_{NUM_AGENTS}_vicuna_wo_feedback.json', 'w') as fp:
                    json.dump(table, fp)
            else:
                with open(f'result_{level}_{NUM_AGENTS}_vicuna.json', 'w') as fp:
                    json.dump(table, fp)

if __name__ == '__main__':
    main()