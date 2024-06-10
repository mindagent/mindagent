import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from levels.utils import convert_to_prompt
from overcooked import World
from utils.llm import chat_llm, prepend_history, rules
from collections import Counter


level='level_10'
file = f'gpt4/result_{level}_2_gpt-4.json'
with open(file, 'r') as f:
    data = json.load(f)
        


alphas = [1.0, 1.5, 2.0, 2.5, 3.0]
for alpha in alphas:
    total = 0
    success = 0
    failed = 0

    env = World(recipe_filename='./assets/recipe.json', task_filename= 'assets/tasks_level_final.json', level='level_10', 
                use_task_lifetime_interval_oracle=True,
                alpha=alpha, beta=2.5, num_agents=4, override_agent=True)

    action_histories = data[str(alpha)]['action_history']
    # print(len(action_histories))

    for eps_id in range(3):
        obs = env.reset()
        for idx, plan in enumerate(action_histories[eps_id]):
            obs, done, info = env.step(plan)

        total += env.success_count + env.failed_count + len(env.task_manager._current_task_list)
        success += env.success_count
        failed += env.failed_count
        print(env.task_manager.accomplished_tasks())

    print(alpha, '\t', 'success: ', success, '\t', 'failed: ', failed, '\t', 'total: ', total)
