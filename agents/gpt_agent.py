import os
import sys
import time
import json
import openai

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from levels.utils import convert_to_prompt
from overcooked import World
from utils.llm import completion_llm, prepend_prompt, rules
import tiktoken

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def main():
    
    model='davinci-003'
    NUM_AGENTS = 2
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

            prompt_base_init = open('./assets/prompt_1.txt', 'r').read().split('###\n')[1]
            history = 6

            num_success = 0
            
            prompt_history = []
            total_action_histories = [] 
            total_action_success_histories = []
            total_prompts = []
            for eps_id in range(max_episode):
                obs = env.reset()
                prompt_base = rules(env) + prompt_base_init +f"There are {env.num_agents} agents available, so you can execute {env.num_agents} actions at a time.\n"
                goal = env.task
                step = 0
                done = False

                prompt = ''
                prompt = prepend_prompt(prompt, prompt_base, verbose=False)
                feedback = '-execution error messages:\n  --  []\n'
                suggestions = '-execution suggestions:\n  --  []\n'
                added_prompt = []

                action_histories = [] 
                action_success_histories = []
                prompt_history = []
                total = 0
                success = 0
                failed = 0

                while True:
                    # update prompt
                    if len(added_prompt) > 2:
                        added_prompt.pop(0)
                        added_prompt.pop(0)
                        prompt = prepend_prompt('', prompt_base, verbose=False)
                        for tmp in added_prompt:
                            prompt = prepend_prompt(prompt, tmp, verbose=False)

                    prompt_tmp =  feedback +  suggestions + convert_to_prompt(obs) + '-action:\n'
                    added_prompt.append(prompt_tmp)

                    prompt = prepend_prompt(prompt, prompt_tmp)

                    if step >= max_steps:
                            break
                    # plan and execute
                    # cnt = num_tokens_from_string(prompt, 'cl100k_base')

                    raw_prompt = convert_to_prompt(obs)
                    prompt_history.append(raw_prompt)
                    
                    plan = completion_llm(prompt)
                    plan = list(filter(lambda x:x, plan.split('\n')))[:2]
                    obs, done, info = env.step(plan)
                    feedback = '-execution error messages:\n  --  ' + str(env.feedback) + '\n'
                    suggestions = '-execution suggestions:\n  --  ' + str(env.suggestions) + '\n'

                    action_histories.append(plan)

                    to_add = ''
                    for i in range(env.num_agents):
                        if i < len(plan):
                            to_add += plan[i] + '\n'

                    prompt = prepend_prompt(prompt, to_add)
                    added_prompt.append(to_add)
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

            with open(f'result_{level}_{NUM_AGENTS}_{model}.json', 'w') as fp:
                json.dump(table, fp)

if __name__ == '__main__':
    main()