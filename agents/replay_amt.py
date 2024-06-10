import argparse
import json
import os
import pickle
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from collections import Counter

from levels.utils import convert_to_prompt
from overcooked import World
from utils.llm import chat_llm, prepend_history, rules


def main(gameplay):
    data = pickle.load(open(args.gameplay, 'rb'))
    env = World(num_agents=data['num_agents'], max_steps=data['world_max_steps'], level=data['level_name'], recipe_filename=data['recipe_filename'], task_filename='assets/tasks_level.json', seed=data['world_seed'], use_task_lifetime_interval_oracle=data['world_use_task_lifetime_interval_oracle'], alpha=data['world_alpha'], beta=data['world_beta'], override_agent=True)
    user_id = data['user_id']
    action_history = data['action_history']
    state_history = data['state_history']

    obs = env.reset()
    history = []
    for idx, (plan, state) in enumerate(zip(action_history, state_history)):
        if args.with_feedback:
            feedback = '-execution error messages:\n  --  []\n'
            suggestions = '-execution suggestions:\n  --  []\n'
        else:
            feedback = ''
            suggestions = ''
        prompt =  feedback +  suggestions + convert_to_prompt(obs) + '-action:\n'
        history = prepend_history(history, prompt)
        to_add = ''
        for i in range(env.num_agents):
            if i < len(plan):
                to_add += plan[i] + '\n'
        history = prepend_history(history, to_add, role='assistant')
        obs, done, info = env.step(plan)

        # ensure the state is the same
        for i in dir(obs):
            if '__' not in i:
                try:
                    assert getattr(obs, i) == getattr(state, i)
                except:
                    print(f'obs and cached obs does not match at step {idx} and key {i}')
                    from IPython import embed; embed()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Replay AMT user gameplay")

    # Add the arguments
    parser.add_argument('--gameplay', metavar='game_play', type=str, help='gameplay file to use')
    parser.add_argument('--with-feedback', action='store_true', help='whether to use feedback')
    # Parse the arguments
    args = parser.parse_args()

    main(args)