import math
from typing import Union, Sequence

import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np

from levels.constants import all_actions, all_objecsts, all_location_types
from overcooked.game import World
from levels.utils import convert_to_prompt, convert_to_state, get_goal_prompt, get_state_prompt
from sentence_transformers import SentenceTransformer
from functools import lru_cache
class NNActionSpaceWrapper(gym.Wrapper):
    """
    Action wrapper to transform native action space to a new space friendly to train NNs
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')
    model.eval()

    prompt_cache = {}

    def __init__(
        self,
        env: Union[World, gym.Wrapper],
    ):
        
        super().__init__(env=env)

        self.all_objecsts = all_objecsts
        self.all_location_types = all_location_types

        num_objects = len(self.all_objecsts)
        num_tools = len(self.all_location_types) 

        self.num_objects = num_objects
        self.num_tools = num_tools

        self.num_agents = env.num_agents
        res = []
        # agent 0
        # 3,  # agent id 0 1 2 
        
        # noop
        # goto: self.num_tools * 3, # tools: noop, storage, blender, chopboard, pan, servingtable, fryer, mixer, pot, steamer,
        # put: self.num_tools * 3, # tools: noop, storage, blender, chopboard, pan, servingtable, fryer, mixer, pot, steamer,
        # activate: self.num_tools * 3, # tools: noop, storage, blender, chopboard, pan, servingtable, fryer, mixer, pot, steamer,
        # get: self.num_tools * 3  + self.num_objects, # tools: noop, storage, blender, chopboard, pan, servingtable, fryer, mixer, pot, steamer,
        # under this contraints get_storage will never be valid, it should be get_storage_something, we use something here to represent the location
        noop_vec = []
        self.agent_offset = (1 + self.num_tools * 3 * 4 + self.num_objects)
        for agent_id in range(self.num_agents):
            res.extend([self.agent_offset * self.num_agents ])
            noop_vec.extend([self.agent_offset * agent_id])

        self.action_space = spaces.MultiDiscrete( res)

        obs_space = env.observation_space
        # obs_space["prompt"] = spaces.Box(low = -np.inf, high=np.inf, shape=(384,))


    def action(self, action: Sequence[int]):
        """
        NN action to game action
        """
        # print('hello', action)
        assert self.action_space.contains(action)

        # ------ parse main actions ------
        # parse agent verb tool and item

        offset = 1
        str_actions = []

        for idx in range(self.num_agents):
            start_idx = idx * offset

            # agent_id = action[start_idx]
            action_number = action[start_idx]
            agent_id =  int(action_number // self.agent_offset)
            action_number = action_number % self.agent_offset
            def parse_action_number(action_number):
                tool_type = None
                tool_id = None
                item = None

                if action_number == 0:
                    verb = all_actions[all_actions.index("noop")]

                elif action_number < self.num_tools * 3 * 1 + 1:
                    verb = all_actions[all_actions.index("goto")]
                    start_idx = (action_number-1)
                    tool_type = all_location_types[start_idx//3]
                    tool_id = start_idx % 3

                elif action_number < self.num_tools * 3 * 2 + 1:
                    verb = all_actions[all_actions.index("put")]
                    start_idx = (action_number-1 - self.num_tools * 3 * 1)
                    tool_type = all_location_types[start_idx//3]
                    tool_id = start_idx % 3

                elif action_number < self.num_tools * 3 * 3 + 1:
                    verb = all_actions[all_actions.index("activate")]
                    start_idx = (action_number-1 - self.num_tools * 3 * 2)
                    tool_type = all_location_types[start_idx//3]
                    tool_id = start_idx % 3

                elif action_number < self.num_tools * 3 * 4 + 1:
                    verb = all_actions[all_actions.index("get")]
                    start_idx = (action_number-1 - self.num_tools * 3 * 3)
                    tool_type = all_location_types[start_idx//3]
                    tool_id = start_idx % 3

                else:
                    verb = all_actions[all_actions.index("get")]
                    start_idx = (action_number-1 - self.num_tools * 3 * 4)
                    tool_type = 'storage'
                    tool_id = 0
                    item = self.all_objecsts[start_idx]

                return verb, tool_type, tool_id, item


            verb, tool_type, tool_id, item = parse_action_number(action_number=action_number)

            if tool_type:
                tool = tool_type + str(tool_id)

            if verb == 'noop':
                str_action = f'noop_agent{agent_id}'
                
            elif verb == 'goto':
                str_action = f'goto_agent{agent_id}_{tool}'

            elif verb == 'get':
                if item:
                    str_action = f'get_agent{agent_id}_{item}_{tool}'
                else:
                    str_action = f'get_agent{agent_id}_{tool}'

            elif verb == 'put':
                if item:
                    str_action = f'put_agent{agent_id}_{item}_{tool}'
                else:
                    str_action = f'put_agent{agent_id}_{tool}'

            elif verb == 'activate':
                str_action = f'activate_agent{agent_id}_{tool}'

            str_actions.append(str_action)
        
        return str_actions

    def reverse_action(self, actions):
        """
        game action to NN action
        """
        nn_actions = []
        for idx, action in enumerate(actions):
            cmd_str = action
            
            cmd_str = cmd_str.split('_')
            
            predicate = cmd_str[0]

            agent = cmd_str[1]
            
            agent = int(agent.replace('agent', ''))
            verb = all_actions.index(predicate)
            tool = 0
            item = 0
            
            start_idx = 1 + (verb-1) * self.num_tools * 3 + self.agent_offset * agent

            if len(cmd_str) == 2:
                # noop
                tool = self.agent_offset * agent
                
            elif len(cmd_str) == 3:
                #goto_agent0_somelocation
                #get_agent0_somelocation
                #put_agnet0_somelocation
                #activate_agnet0_somelocation

                tool_id = int(cmd_str[-1][-1])
                
                tool = self.all_location_types.index(cmd_str[-1][:-1]) * 3  + tool_id + start_idx
                
            elif len(cmd_str) == 4:
            
                #get_agent0_something_somelocation
                #put_agnet0_something_somelocation
                tool_id = int(cmd_str[-1][-1])
                if cmd_str[-1][:-1] == 'storage':
                    start_idx = 1 + 4 * self.num_tools * 3 + self.agent_offset * agent
                    tool = self.all_objecsts.index(cmd_str[-2]) + start_idx
                else:
                    tool = self.all_location_types.index(cmd_str[-1][:-1]) * 3 + tool_id + start_idx
                item = self.all_objecsts.index(cmd_str[2])
            
            nn_action = [tool]
            nn_actions.append(nn_action)

        # while len(nn_actions) < self.num_agents:
        #     cur_agnet = len(nn_actions)
        #     nn_actions.append([cur_agnet, 0, 0, 0])
        from itertools import chain
        nn_actions = list(chain(*nn_actions))
        return nn_actions
       

    def reset(self, **kwargs):
        
        self.rewards = []
        if kwargs['options']:
            task_name = kwargs['options']['task_name']
            obs = self.env.reset(task_name = task_name)
        else:
            obs = self.env.reset()
        if self.env.use_state_observation:
            obs = convert_to_state(obs)
        else:
            goal_prompt = get_goal_prompt(obs)
            state_prompt = get_state_prompt(obs)
            goal_encoding = self.encode(goal_prompt)
            state_encoding = self.encode(state_prompt)

            obs = np.hstack((goal_encoding, state_encoding))
        
        
        return obs, {}
    
    @lru_cache(maxsize=2560000)
    def encode(self, input_prompt):
        return NNActionSpaceWrapper.model.encode(input_prompt)

    def step(self, action: Sequence[int]):
        overcooked_action = self.action(action)
        obs, done, info = self.env.step(overcooked_action)
        reward = info['reward']
        # obs = convert_to_prompt(obs)
        if self.env.use_state_observation:
            obs = convert_to_state(obs)
        else:
            # prompt = convert_to_prompt(obs)
            goal_prompt = get_goal_prompt(obs)
            state_prompt = get_state_prompt(obs)

            goal_encoding = self.encode(goal_prompt)
            state_encoding = self.encode(state_prompt)
            # if prompt in NNActionSpaceWrapper.prompt_cache:
            #     obs = NNActionSpaceWrapper.prompt_cache[prompt]
            # else:
            #     obs = NNActionSpaceWrapper.model.encode(prompt)
            #     NNActionSpaceWrapper.prompt_cache[prompt] = obs
            #     if len(NNActionSpaceWrapper.prompt_cache) % 1000 == 0:
            #         print('cache size: ', len(NNActionSpaceWrapper.prompt_cache))
            obs = np.hstack((goal_encoding, state_encoding))

        self.rewards.append(reward)

        info['just_success'] = done
        if self.env.time_step == self.env.max_steps:
            done = True
        else:
            done = False

        # gymnasium require this
        truncated = False
        return obs, reward, done, truncated, info