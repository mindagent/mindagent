from typing import Any, Union, Optional

import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np

from levels.constants import all_actions, all_objecsts, all_location_types

from overcooked.game import World
from levels.utils import convert_to_prompt

class ARMasksWrapper(gym.ObservationWrapper):
    def __init__(
        self,
        env: Union[World, gym.Wrapper]
    ):

        assert isinstance(
            env.action_space, spaces.MultiDiscrete
        ), "please use this wrapper with `NNActionSpaceWrapper!`"
        assert (
            len(env.action_space.nvec) == env.num_agents
        ), "please use this wrapper with `NNActionSpaceWrapper!`"
        super().__init__(env=env)

        self.all_actions = all_actions

        self.all_objecsts = all_objecsts
        self.all_location_types = all_location_types

        self.num_objects = len(self.all_objecsts)
        self.num_tools = len(self.all_location_types) 
        self.num_actions = len(self.all_actions)
    
        obs_space = env.observation_space
        # total observations = # agents * (# actions + num_tools * (goto + get + put + activate) + #objects)
 
        # obs_space["masks"] = spaces.Box(low=0, high=1, shape=(self.num_obs,), dtype=bool)
        
        # obs_space["masks"] = spaces.Dict(
        #     {
        #         # agent 0
        #         "verb_agent0": spaces.Box(
        #             low=0, high=1, shape=(self.num_actions,), dtype=bool
        #         ),

        #         "goto_tool_agent0": spaces.Box(
        #             low=0, high=1, shape=(self.num_tools, 1), dtype=bool
        #         ),

        #         "get_tool_agent0": spaces.Box(
        #             low=0, high=1, shape=(self.num_tools, 1), dtype=bool
        #         ),

        #         "put_tool_agent0": spaces.Box(
        #             low=0, high=1, shape=(self.num_tools, 1), dtype=bool
        #         ),

        #         "activate_tool_agent0": spaces.Box(
        #             low=0, high=1, shape=(self.num_tools, 1), dtype=bool
        #         ),

        #         "get_object_agent0": spaces.Box(
        #             low=0, high=1, shape=(self.num_objects, 1), dtype=bool
        #         ),

        #         # agent 1
        #         "verb_agent1": spaces.Box(
        #             low=0, high=1, shape=(self.num_actions,), dtype=bool
        #         ),

        #         "goto_tool_agent1": spaces.Box(
        #             low=0, high=1, shape=(self.num_tools, 1), dtype=bool
        #         ),

        #         "get_tool_agent1": spaces.Box(
        #             low=0, high=1, shape=(self.num_tools, 1), dtype=bool
        #         ),

        #         "put_tool_agent1": spaces.Box(
        #             low=0, high=1, shape=(self.num_tools, 1), dtype=bool
        #         ),

        #         "activate_tool_agent1": spaces.Box(
        #             low=0, high=1, shape=(self.num_tools, 1), dtype=bool
        #         ),

        #         "get_object_agent1": spaces.Box(
        #             low=0, high=1, shape=(self.num_objects, 1), dtype=bool
        #         ),


        #         # agent 2
        #         "verb_agent2": spaces.Box(
        #             low=0, high=1, shape=(self.num_actions,), dtype=bool
        #         ),

        #         "goto_tool_agent2": spaces.Box(
        #             low=0, high=1, shape=(self.num_tools, 1), dtype=bool
        #         ),

        #         "get_tool_agent2": spaces.Box(
        #             low=0, high=1, shape=(self.num_tools, 1), dtype=bool
        #         ),

        #         "put_tool_agent2": spaces.Box(
        #             low=0, high=1, shape=(self.num_tools, 1), dtype=bool
        #         ),

        #         "activate_tool_agent2": spaces.Box(
        #             low=0, high=1, shape=(self.num_tools, 1), dtype=bool
        #         ),

        #         "get_object_agent2": spaces.Box(
        #             low=0, high=1, shape=(self.num_objects, 1), dtype=bool
        #         ),
                
        #     }
        # )

    def get_mask(self):
        available_actions = self.env.available_actions(return_struct=True)
         #  struct_actions = {
        #         'noop': noops,
        #         'goto': gotos,
        #         'put': puts,
        #         'get': gets,
        #         'activate': activates
        #     }
        
      
        
        self.agent_offset = 1 + self.num_tools * 3 * 4 + self.num_objects
        masks = np.zeros(self.agent_offset * 2)
        for agent_id,action in enumerate(available_actions):

            # goto: self.num_tools * 3, # tools: noop, storage, blender, chopboard, pan, servingtable, fryer, mixer, pot, steamer,
            # put: self.num_tools * 3, # tools: noop, storage, blender, chopboard, pan, servingtable, fryer, mixer, pot, steamer,
            # activate: self.num_tools * 3, # tools: noop, storage, blender, chopboard, pan, servingtable, fryer, mixer, pot, steamer,
            # get: self.num_tools * 3  + self.num_objects, # tools: noop, storage, blender, chopboard, pan, servingtable, fryer, mixer, pot, steamer,
            # under this contraints get_storage will never be valid, it should be get_storage_something, we use something here to represent the location
            
            # noop_mask = np.array([1])
            # goto_masks = np.zeros(self.num_tools * 3)
            
            # put_masks = np.zeros(self.num_tools * 3)
            # activate_masks = np.zeros(self.num_tools * 3)
            # get_masks = np.zeros(self.num_tools * 3 + self.num_objects)

            # object_masks = np.zeros(self.num_objects)

            
            

            #action primitives: noop, goto, get, put, activate
            #noop
            masks[agent_id * self.agent_offset] = 1

            # goto
            for loc in action['goto']:
                verb = self.all_actions.index('goto')
                start_idx = 1 + (verb-1) * self.num_tools * 3 + self.agent_offset * agent_id
                loc_id = int(loc[-1][-1])
                loc_name = loc[-1][:-1]
                # self.all_location_types.index(cmd_str[-1][:-1]) * 3 + tool_id + start_idx
                loc_idx = self.all_location_types.index(loc_name)*3 + loc_id +  start_idx
                masks[loc_idx] = 1

            # get
           
            for loc in action['get']:
                verb = self.all_actions.index('get')
                start_idx = 1 + (verb-1) * self.num_tools * 3 + self.agent_offset * agent_id
                loc_id = int(loc[-1][-1])
                loc_name = loc[-1][:-1]
                
                # if get from storage, need to specify items
                if loc_name == 'storage':
                    item = loc[1]
                    # masks[self.all_objecsts.index(item)] = 1 
                    start_idx = 1 + 4 * self.num_tools * 3 + self.agent_offset * agent_id
                    item = loc[-2]
                    loc_idx = self.all_objecsts.index(item) + start_idx
                    masks[loc_idx] = 1
                else:
                    loc_idx = self.all_location_types.index(loc_name)*3 + loc_id +  start_idx
                    masks[loc_idx] = 1
            
            #put 
            for loc in action['put']:
                verb = self.all_actions.index('put')
                start_idx = 1 + (verb-1) * self.num_tools * 3 + self.agent_offset * agent_id
                loc_id = int(loc[-1][-1])
                loc_name = loc[-1][:-1]
                
                loc_idx = self.all_location_types.index(loc_name)*3 + loc_id +  start_idx
                masks[loc_idx] = 1

            #activate
            for loc in action['activate']:
                verb = self.all_actions.index('activate')
                start_idx = 1 + (verb-1) * self.num_tools * 3 + self.agent_offset * agent_id
                loc_id = int(loc[-1][-1])
                loc_name = loc[-1][:-1]
                
                loc_idx = self.all_location_types.index(loc_name)*3 + loc_id +  start_idx
                masks[loc_idx] = 1

        masks = np.array([masks, masks])
        return masks
    
    def observation(self, observation: dict[str, Any]):
        return observation