import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from levels.utils import convert_to_prompt
from overcooked import World
from utils.llm import chat_llm, prepend_history, rules
from collections import Counter

directories = ['3agents_07_08']
levels = ['level_3'] #'level_3', 'level_4', 'level_5']
errors = []
for level in levels:
    for directory in directories:
        files =  os.listdir(directory)
        datas = []
        for file in files:
            if level in file:
                with open(f'{directory}/{file}', 'r') as f:
                    data = json.load(f)
            

    alpha = 1.0
    env = World(recipe_filename='./assets/recipe.json', level=level, 
                use_task_lifetime_interval_oracle=True,
                alpha=alpha, beta=2.5, num_agents=3, override_agent=True)

    action_histories = data[str(alpha)]['action_history']
    # print(len(action_histories))
    
    for eps_id in range(3):
        obs = env.reset()
        cnt = 0
        action_success_histories = []
        previous_plan = None
        found_waste = False
        for idx, plan in enumerate(action_histories[eps_id]):
            prompt_before = convert_to_prompt(obs)
            obs, done, info = env.step(plan)
            print('++++++++++')
            print(prompt_before)
            print('plan: ', plan)
            prompt = convert_to_prompt(obs)
            print(prompt)
            print('-----------')
            # gt_prompt = data[str(alpha)]['prompt_history'][eps_id][idx+1]
            # action_success_histories.append(info['action_success'])
            if env.feedback:
            # if 'waste' in prompt and not found_waste: 
            #     found_waste = True  
            #     print('prompt before ++++++++ ')
            #     print(prompt_before)
            #     print(" plan: ", previous_plan )
            #     print('prompt after: ', prompt)
            #     print('----------')
                errors.extend(env.feedback)

            # if 'waste' not in prompt:
            #     found_waste = False
            
            previous_plan = plan
        
counter = Counter(errors)
print(counter)
