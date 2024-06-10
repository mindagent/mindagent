# all the predicates and states
import copy
import json
import os
import random
import warnings
from collections import Counter, defaultdict
from typing import List, Optional
from uuid import uuid1

import gymnasium.spaces as spaces
import numpy as np

from levels.constants import (ItemName, LocationType, StepReturnType,
                              all_objecsts, base_ingridients)
from levels.utils import (compute_dependency, compute_lifetime_task_intervals,
                          extract_agent_id, filter_recipe)


class Location:
    def __init__(self, loc_type : LocationType, name:str,  occupied_time : int, capacity: int, need_watch: bool) -> None:
        self.type : LocationType = loc_type
        self.name = name
        self.content : Optional[Item] = None
        self.refresh_time = occupied_time
        self.is_occupied : int = 0
        self.need_watch = need_watch
        self.capacity : int = capacity

        # TODO(jxma): you cannot pick from serving table and you can always pick from storage and chopboard
        self.pickable : bool = False
        if self.type == LocationType.STORAGE or self.type == LocationType.CHOPBOARD:
            self.pickable = True

    def __repr__(self):
        return self.name
    def add(self, item):
        if self.content is None:
            self.content = item
        else:
            # for serving table, we do not merge the same item as a task might be "two porkMeatCake"
            merge_item = False if self.type == LocationType.SERVINGTABLE else True
            self.content.add(item, merge_item=merge_item)

    def toggle_pickable(self):
        # TODO(jxma): you cannot pick from serving table and you can always pick from storage
        if self.type == LocationType.SERVINGTABLE:
            self.pickable = False
        elif self.type == LocationType.STORAGE or self.type == LocationType.CHOPBOARD:
            self.pickable = True
        else:
            self.pickable = not self.pickable

    @property
    def quantity(self):
        if self.content:
            return self.content.quantity
        else:
            return 0

class Item():
    def __init__(self, item ) -> None:
        self.name = item

    def add(self, item, merge_item=True):
        contents = self.name.split('&')
        if merge_item and item.name in contents:
            return
        self.name += '&' + item.name

    def all_content(self):
        return self.name.split('&')

    @property
    def quantity(self):
        return len(self.all_content())

    def cook(self, location: Location, recipe):
        # do the cook magic (inc. chop, steam, pan fry, blend, etc) here
        # based on recipe
        all_ingredients = self.all_content()
        dish = recipe.make(location, all_ingredients)

        return dish

    def __eq__(self, other):
        if isinstance(other, Item):
            return self.name == other.name
        return False

# all the recipes
class RECIPE:
    def __init__(self, recipe: dict):
        self.recipe = recipe
        self.raw_recipe = copy.deepcopy(recipe)
        self.process_recipe()

    def process_recipe(self):
        tmp = {}
        for k, v in self.recipe.items():
            tmp['_'.join(sorted(v['ingredients']) + [v['location']])] = k
        self.recipe = tmp

    def make(self, location: Location, all_ingredients: List[str]) -> Item:
        sorted_ing = '_'.join(sorted(all_ingredients) + [location.type.value])
        # TODO: in the actual game, cooking will be impossible for wrong ingredients
        # but here we have them cooked into waste

        if sorted_ing not in self.recipe:
            return Item(ItemName.waste)
        else:
            return Item(ItemName.from_str(self.recipe[sorted_ing]))


class TaskManagerReturnType:
    def __init__(self,
                 task_just_success = [],
                 task_just_success_location = [] ,
                 game_over = False,
                 just_success_remaining_time = None,
                 just_failed = False) -> None:
        # a list of the name of the tasks just become successful at this step
        self.task_just_success = task_just_success
        # a list of the location of the tasks just become successful at this step
        self.task_just_success_location = task_just_success_location
        self.game_over = game_over
        self.just_success_remaining_time = just_success_remaining_time
        self.just_failed = just_failed

class TaskManager:

    def __init__(self,
                 rng,
                 task_def=None,
                 # how many steps a new task will be added, -1 means no new task will be added
                 task_interval=-1,
                 # -1 means infinite
                 task_lifetime=-1,
                 num_agents = 1,
                 # whether to use the lifetime and task interval oracle
                 use_task_lifetime_interval_oracle = False,
                 # maximum number of tasks in the current task list
                 max_num_tasks = 6, alpha=None, beta=None):
        self._task_def = task_def
        self.num_agents = num_agents
        self.rng = rng
        self._all_tasks = list(self._task_def['task_list'].keys())

        if use_task_lifetime_interval_oracle:
            # rewrite task def
            assert task_interval == -1
            assert task_lifetime == -1
            task_intervals = []
            for task in self._task_def['task_list'].keys():
                lifetime, interval = compute_lifetime_task_intervals(task, num_agents, alpha=alpha, beta=beta)
                self._task_def['task_list'][task]['task_lifetime'] = lifetime
                task_intervals.append(interval)
            # TODO(jxma): tentatively, we use the max interval
            self._task_def['task_interval'] = max(task_intervals)

        self._max_num_tasks = max_num_tasks
        self._task_interval = task_interval
        if "task_interval" in self._task_def:
            self._task_interval = self._task_def['task_interval']
        self._task_lifetime = task_lifetime
        self._current_task_list = []
        self._current_task_lifetime_list = []
        self._timer = 0
        self._accomplished_task_list = []
        self._tools_mapping = {}

    def _random_task(self):
        return self.rng.choice(self._all_tasks)

    def reset(self, first_task=None, tools_mapping=None):
        #FIXME(jxma): tentatively use strict check
        assert tools_mapping is not None
        self._tools_mapping = tools_mapping
        self._all_tasks = list(self._task_def['task_list'].keys())
        self._current_task_list = []
        self._current_task_lifetime_list = []
        # four types of memory to help shape the reward
        self.base_picked = { key: 0 for key in all_objecsts }
        self.loc_satisfied = {}
        self.intermediate_cooked = {key: 0 for key in all_objecsts}
        self.hold_reward_given = {key: 0 for key in all_objecsts}
        self.reward_memory = {'agent_'+str(key): None for key in range(self.num_agents)}
        self.reward_memory.update({
          key: [] for key, value in self._tools_mapping.items()
        })
        if first_task:
            assert first_task in self._all_tasks
            self._current_task_list.append(first_task)
        else:
            first_task = self._random_task()
            self._current_task_list.append(first_task)

        all_required_items, tools, reaward_mappings, mappings = compute_dependency(first_task, return_mapping=True)
        if len(mappings) > 0:
            self.update_reward_shaping_memories(all_required_items, mappings)

        if 'task_lifetime' in self._task_def['task_list'][self._current_task_list[-1]]:
            self._current_task_lifetime_list.append(
                self._task_def['task_list'][self._current_task_list[-1]]['task_lifetime'])
        else:
            self._current_task_lifetime_list.append(self._task_lifetime)

        self._timer = 0
        self._accomplished_task_list = []

    def update_reward_shaping_memories(self, all_required_items, mappings):
        # update the four memories
        for item in all_required_items:
            if item in base_ingridients:
                self.base_picked[item] += 1
            elif item != all_required_items[-1]:
                self.intermediate_cooked[item] += 1
            else:
                self.hold_reward_given[item] += 1
                self.intermediate_cooked[item] += 1

        for loc_type , list_products in mappings.items():
            for products in list_products:
                if (loc_type, '&'.join(sorted(products)) ) not in self.loc_satisfied:
                    self.loc_satisfied[(loc_type, '&'.join(sorted(products)) ) ] = 1
                else:
                    self.loc_satisfied[(loc_type, '&'.join(sorted(products)) ) ] += 1

    def tick(self):
        self._timer += 1
        # lifetime decrease
        for i in range(len(self._current_task_lifetime_list)):
            # in case the lifetime is not -1, which means infinite
            if self._current_task_lifetime_list[i] != -1:
                self._current_task_lifetime_list[i] -= 1

        # add a new task
        if self._timer == self._task_interval:
            if len(self._current_task_list) < self._max_num_tasks:
                added_task = self._random_task()
                self._current_task_list.append(added_task)
                all_required_items, tools,  reaward_mappings, mappings = compute_dependency(added_task, return_mapping=True)
                if len(mappings) > 0:
                    self.update_reward_shaping_memories(all_required_items, mappings)
                if 'task_lifetime' in self._task_def['task_list'][self._current_task_list[-1]]:
                    self._current_task_lifetime_list.append(
                        self._task_def['task_list'][self._current_task_list[-1]]['task_lifetime'])
                else:
                    self._current_task_lifetime_list.append(self._task_lifetime)
            self._timer = 0

        # sort all tasks based on the lifetime
        tmp = sorted(zip(self._current_task_lifetime_list, self._current_task_list))
        self._current_task_list = [x[1] for x in tmp]
        self._current_task_lifetime_list = [x[0] for x in tmp]


    def compute_reward(self, world_state, agents):
        # print('self.intermediate_cooked: ', self.intermediate_cooked)
        if len(self._current_task_list) == 0:
            return 0

        reward = 0

        for agent_id, agent in enumerate(agents):
            if agent.holding and agent.holding.name != 'waste' and self.base_picked[agent.holding.name] > 0:
                if agent.holding.name not in self.reward_memory[f'agent_{agent_id}']:
                    self.reward_memory[f'agent_{agent_id}'].append(agent.holding.name)
                    reward += 0.1
                    self.base_picked[agent.holding.name] -= 1

        for loc_name, loc in world_state.items():
            loc_type = loc_name[:-1]
            if loc.content:
                contents = sorted(loc.content.name.split('&'))
                tmp = (loc_type, '&'.join(contents))

                if tmp in  self.loc_satisfied and self.loc_satisfied[tmp] > 0:
                    if loc.content.name not in self.reward_memory[loc_name]:
                        self.loc_satisfied[tmp] -= 1
                        reward += 0.5
                        self.reward_memory[loc_name].append(loc.content.name)

        for loc_name, loc in world_state.items():
            if loc.content:
                if  loc.content.name in self.intermediate_cooked and self.intermediate_cooked[loc.content.name] > 0:
                    if loc.content.name not in self.reward_memory[loc_name]:
                        reward += 1.0
                        self.intermediate_cooked[loc.content.name] -= 1
                        self.reward_memory[loc_name].append(loc.content.name)

        for agent_id, agent in enumerate(agents):
            if agent.holding and agent.holding.name != 'waste' and self.hold_reward_given[agent.holding.name] > 0:
                if agent.holding.name in self.reward_memory[f'agent_{agent_id}']:
                    self.reward_memory[f'agent_{agent_id}'].append(agent.holding.name)
                    reward += 0.5
                    self.hold_reward_given[agent.holding.name] -= 1

        return reward

    def check_task_success(self, world_state):
        # Return: TaskManagerReturnType

        task_just_success = []
        task_just_success_location = []
        game_over = False
        just_failed = False
        rmv_id = []
        failed_id = []

        # if any of the current task reach the lifetime, remove it from the game and mark `just_failed` as True
        for ind in range(len(self._current_task_lifetime_list)):
            if self._current_task_lifetime_list[ind] == 0:
                rmv_id.append(ind)
                failed_id.append(ind)
                just_failed = True

        # Note: here the we choose to remove the first task from the list (when there are multiple tasks of the same type).
        # check if any of the current task is accomplished
        ingredients = [self.task_def['task_list'][i]['ingredients'] for i in self._current_task_list]
        locations = [self.task_def['task_list'][i]['location'] for i in self._current_task_list]

        all_loc_counter = {}
        for loc_name, loc in world_state.items():
            if loc.content:
                loc_counter = Counter(loc.content.name.split('&'))
            else:
                loc_counter = None
            all_loc_counter[loc_name] = loc_counter
        for ind, (target_ing, target_loc) in enumerate(zip(ingredients, locations)):
            target_counter = Counter(target_ing)
            for loc_name, loc in world_state.items():
                if loc.type.value == target_loc:
                    # all item in loc_counter is not less than requirement
                    if all_loc_counter[loc_name] and ind not in failed_id:
                        if all(all_loc_counter[loc_name][key] >= target_counter[key] for key in target_counter):
                            # remove these items from all_loc_counter
                            for k, v in target_counter.items():
                                all_loc_counter[loc_name][k] -= v
                            task_just_success.append(self._current_task_list[ind])
                            task_just_success_location.append(loc_name)
                            rmv_id.append(ind)
                            break

        # remove the accomplished task and failed task
        for i in rmv_id:
            if i not in failed_id:
                self._accomplished_task_list.append(self._current_task_list[i])

        self._current_task_list = [x for i, x in enumerate(self._current_task_list) if i not in rmv_id]
        just_success_reamaning_time = [x for i, x in enumerate(self._current_task_lifetime_list) if i in rmv_id and i not in failed_id]
        self._current_task_lifetime_list = [x for i, x in enumerate(self._current_task_lifetime_list) if i not in rmv_id]

        # sort all tasks based on the lifetime
        tmp = sorted(zip(self._current_task_lifetime_list, self._current_task_list))
        self._current_task_list = [x[1] for x in tmp]
        self._current_task_lifetime_list = [x[0] for x in tmp]

        return TaskManagerReturnType(task_just_success, task_just_success_location, game_over, just_success_reamaning_time, just_failed)

    # a list of task name and its current lifetime
    def current_tasks(self):
        return list(zip(self._current_task_list, self._current_task_lifetime_list))

    # a list of all tasks have been accomplished
    def accomplished_tasks(self):
        return copy.deepcopy(self._accomplished_task_list)

    @property
    def task_def(self):
        return self._task_def


class World:
    metadata = {"render.modes": ["text"]}
    def __init__(self,
                 num_agents=2,
                 max_steps=60, # assume every action is 4 seconds, and we have in total 4 minutes
                 recipe_filename='assets/recipe.json',
                 task_filename='assets/tasks_level.json',
                 task_interval=-1,
                 task_lifetime=-1,
                 level='level_1',
                 use_task_lifetime_interval_oracle=False,
                 max_num_tasks=6,
                 seed=0,
                 use_state_observation = False, alpha=None, beta=None, override_agent=False,
                 play_mode = 'gpt_agent') -> None:
        # play_mode: standard, gpt_agent, play_with_human, random_agent 
        self.play_mode = play_mode
        assert self.play_mode in ['gpt_agent', 'play_with_human', 'random_agent']
        
        self.use_task_lifetime_interval_oracle = use_task_lifetime_interval_oracle
        self.alpha = alpha
        self.beta = beta

        self.seed = seed
        self.rng = random.Random(seed)
        self.time_step = 0
        self.recipe_filename = recipe_filename
        self.use_state_observation = use_state_observation
        if self.use_state_observation:
            self.observation_space = spaces.Box(low = -np.inf, high=np.inf, shape=(64,))
        else:
            self.observation_space = spaces.Box(low = -np.inf, high=np.inf, shape=(384*2,))

        with open(task_filename, 'r') as f:
            self.tasks_def = json.load(f)
        if level is not None:
            self.tasks_def = self.tasks_def[level]
            self.level = level
        else:
            self.tasks_def = self.tasks_def['level_1']
            self.level = 'level_1'
        # TODO(jxma): we now use num_agents in the task definition by default
        if 'num_agents' in self.tasks_def:
            self.num_agents = self.tasks_def['num_agents']
        else:
            self.num_agents = num_agents

        if override_agent:
            self.num_agents = num_agents

        if 'max_steps' in self.tasks_def:
            self.max_steps = self.tasks_def['max_steps']
        else:
            self.max_steps = max_steps

        self.task_manager = TaskManager(
                self.rng,
                self.tasks_def,
                task_interval,
                task_lifetime,
                num_agents=num_agents,
                use_task_lifetime_interval_oracle=use_task_lifetime_interval_oracle,
                max_num_tasks=max_num_tasks,
                alpha=alpha,
                beta=beta)

        self.action_history = []
        self.state_history = []
        self.tool_cnt = defaultdict(int)

        self._episode_info = TaskManagerReturnType()

        self.feedback = []
        self.previous_actions= []


    def kitchen_def(self):
        #TODO (stg): return the list of cooking tools
        if 'kitchen' in self.tasks_def:
            return self.tasks_def['kitchen']
        else:
            raise ValueError(f'kitchen not found in task definition')

    @property
    def task(self):
        # TODO(jxma): legacy, return the first task of the current task list
        return self.task_manager._current_task_list[0]

    def setup_tool(self, tool):
        from levels.constants import (capacity, need_watch, occupied_time,
                                      type_table)
        tool_capacity = capacity[tool]
        tool_occupied_time = occupied_time[tool]
        tool_need_watch = need_watch[tool]

        location = Location(type_table[tool], f"{tool}{self.tool_cnt[tool]}" ,tool_occupied_time, tool_capacity, tool_need_watch)
        self.tool_cnt[f'{tool}'] += 1

        return location

    def load_level(self):

        # TODO(jxma): legacy, compute kitchen from the task, will be removed.
        # from levels.utils import compute_dependency
        # topological_order, level = compute_dependency(self.task)

        level = self.kitchen_def()

        locations = []
        for item in level:
            location = self.setup_tool(item)
            locations.append(location)

        self.name_mapping = { loc.name: loc for loc in locations }

        for key, value in self.name_mapping.items():
            setattr(self, key, value)

        self.agents : List[Agent] = []
        for idx in range(self.num_agents):
            self.agents.append(Agent(idx, self))

        # TODO(jxma): only relevant recipe is valid
        with open(self.recipe_filename, 'r') as f:
            tmp = json.load(f)
        # tmp = json.load(open(self.recipe_filename, 'r'))
        tmp = filter_recipe(tmp, list(self.tasks_def['task_list'].keys()))
        self.recipe = RECIPE(tmp)

    def done(self):
        # Return TaskManagerReturnType:
        # {
        #     'accomplished_task': a list of accomplished tasks,
        #     'game_over': bool,
        #     'task_just_success': a list of the tasks just become successful at this step
        #       (will become empty again at the next step)
        #     'task_just_success_location': a list of the locations of the tasks just become successful at this step
        #       (will become empty again at the next step)
        # }

        task_manager_return = self.task_manager.check_task_success(self.name_mapping)
        # for k, v in self.name_mapping.items():
        #     print(k, v.content)
        # print(task_manager_return.task_just_success)

        # TODO(jxma): if any task just success at this step, remove the needed items from the corresponding locations
        # otherwise do nothing (this aligns with the original game)

        # if task_manager_return.task_just_success:
        #     for loc_name in self.name_mapping:
        #         if loc_name in task_manager_return.task_just_success_location:
        #             self.name_mapping[loc_name].content = None

        for task_name, loc_name in zip(task_manager_return.task_just_success, task_manager_return.task_just_success_location):
            assert self.name_mapping[loc_name].content is not None
            current_content= self.name_mapping[loc_name].content.name.split('&')
            for ing in self.task_manager.task_def['task_list'][task_name]['ingredients']:
                # print(ing, current_content)
                current_content.remove(ing)
            if current_content:
                self.name_mapping[loc_name].content.name = '&'.join(current_content)
            else:
                self.name_mapping[loc_name].content = None

        # if the maximum game time has reached, the game is over anyway.
        if self.time_step == self.max_steps:
            task_manager_return.game_over = True

        return task_manager_return

    def validate_n_parse(self, cmd_strs: List[str]):
        # validate
        # return True, predicate, args
        # cmd_str: goto_agent0_storage
        # noop_agent0
        # activate_agent0_blender
        # get_agent0_flour_storage
        # put_agent0_flour_blender
        predicates = []
        args = []
        for cmd_str in cmd_strs:
            action = cmd_str
            cmd_str = cmd_str.split('_')
            if len(cmd_str) < 2 or len(cmd_str) > 4:
                # print('not enough arguments')
                self.feedback.append(f'not enough arguments for action {action}')
                return False, None, None, None

            predicate = cmd_str[0]
            if predicate not in ['noop', 'goto', 'get', 'put', 'activate']:
                self.feedback.append(f'{predicate} is not in the list of supported actions')
                return False, None, None, None

            agent = cmd_str[1]
            try:
                agent = int(agent.replace('agent', ''))
            except:
                self.feedback.append(f'agent id not found')
                return False, None, None, None

            if len(cmd_str) == 2:
                # noop
                location = 'none'
                arg = [agent]
            elif len(cmd_str) == 3:
                #goto_agent0_somelocation
                #get_agent0_somelocation
                #put_agnet0_somelocation
                #activate_agnet0_somelocation

                #TODO:
                # need to test if the object can be activated

                location = cmd_str[-1]
                if location not in self.name_mapping:
                    # print('invalid location')
                    self.feedback.append(f'for agent{agent}, tool {location} is not in the current game level')
                    return False, None, None, None

                if predicate == 'get' and location == 'storage':
                    self.feedback.append(f'for agent{agent}, need to specify the item to get from storage')
                    return False, None, None, None

                arg = [agent, location]
            elif len(cmd_str) == 4:
                if predicate != 'get' and predicate != 'put':
                    return False, None, None, None
                #get_agent0_something_somelocation
                #put_agnet0_something_somelocation
                item = cmd_str[2]
                location = cmd_str[3]
                if location not in self.name_mapping:
                    # print('invalid location')
                    self.feedback.append(f'for agent{agent}, tool {location} is not in the current game level')
                    return False, None, None, None
                arg = [agent, item,location]


            predicates.append(predicate)
            args.append(arg)

        # agents must be differnt
        agents = [arg[0] for arg in args]
        if len(agents) != len(set(agents)):
            # print("duplicate agents")
            self.feedback.append(f'agent ids cannot be the same')
            return False, None, None, None


        # make sure no conflict in actions
        # get only if
        valid_predicates = []
        valid_args = []

        # TODO(jxma): avialable_actions won't be used as filtering as now
        # actions of differnt agents will be execed sequentially, therefore some
        # actions may become valid once the previous action is execed.
        ignored_actions = [False for _ in range(len(predicates))]

        # ignored_actions = []
        # avail_actions = self.available_actions(return_struct=True)
        # for predicate, arg in zip(predicates, args):
        #     agnet_id = int(arg[0])
        #     if  arg in avail_actions[agnet_id][predicate]:
        #         # valid_predicates.append(predicate)
        #         # valid_args.append(arg)
        #         ignored_actions.append(False)
        #     else:
        #         ignored_actions.append(True)

        # Initialize a list to store tuples of (predicate, arg, ignored_action)
        actions = list(zip(predicates, args, ignored_actions))

        # Separate the sorted lists back into predicates, args, and ignored_actions
        predicates, args, ignored_actions = zip(*actions)

        # Convert the sorted lists to regular lists
        predicates = list(predicates)
        args = list(args)
        ignored_actions = list(ignored_actions)

        return True, predicates, args, ignored_actions

    def available_actions(self, return_struct=False):
        from levels.constants import base_ingridients
        all_actions = []
        all_struct_actions = []
        for i in range(self.num_agents):
            noops = []
            gotos = []
            gets = []
            puts = []
            activates = []
            actions = []

            # noop
            # agent: not occupied
            if not self.agents[i].is_occupied:
                noops.append([i])
                actions.append(f"noop")

            # goto
            # agent: not occupied; not at that location already
            if not self.agents[i].is_occupied:
                for loc_name, loc in self.name_mapping.items():
                    if not self.agents[i].location.name == loc_name:
                        gotos.append([i, loc_name])
                        actions.append(f"goto {loc_name}")

            # get
            # agent: not holding anything, not occupied, at the location
            # location: not occupied, not empty
            #   if location is storage, can get anything
            #   if location is not storage, can get only if there is something

            # FIXED:  since it's possbile to get right after the first agent put in, so there is
            # no longer a requirement on the if the location is empty.
            # you cannot get anything from serving table
            if (not self.agents[i].holding) and (not self.agents[i].is_occupied):
                for loc_name, loc in self.name_mapping.items():
                    if self.agents[i].location.name == loc_name and not loc.is_occupied:
                        if loc_name.startswith("storage"):
                            for item in base_ingridients:
                                gets.append([i,item, loc_name])
                                actions.append(f"get {item} {loc_name}")
                        if not loc.content or loc_name.startswith("servingtable"):
                            continue
                        else:
                            gets.append([i, loc_name])
                            actions.append(f"get {loc_name}")

            # put
            # agent: not holding anything, not occupied, at the location
            # location: not occupied
            # we don't allow put action that will lead to waste, unless the tool is empty
            if ( self.agents[i].holding ) and (not self.agents[i].is_occupied) :
                for loc_name, loc in self.name_mapping.items():
                        if self.agents[i].location.name == loc_name and not loc.is_occupied:
                            if self.agents[i].holding and loc.content:
                                if not loc.name.startswith("servingtable") and not loc.name.startswith("storage"):
                                    tmp = copy.deepcopy(loc.content)
                                    tmp.add(self.agents[i].holding)
                                    if tmp.cook(loc, self.recipe).name == 'waste':
                                        continue
                            puts.append([i, loc_name])
                            actions.append((f"put {loc_name}"))

            # activate
            # agent: not holding anything, not occupied, at the location
            # location: not empty, not occupied, not storage or servingtable
            # FIXED:  since it's possbile to activate right after the first agent put in, so there is
            # no longer a requirement on the if the location is empty.
            # activate action that will lead to waste is not allowed
            if (not self.agents[i].holding) and (not self.agents[i].is_occupied):
                for loc_name, loc in self.name_mapping.items():
                    if loc.type in [LocationType.SERVINGTABLE, LocationType.STORAGE]:
                        continue
                    if self.agents[i].location.name == loc_name and (not loc.is_occupied):
                        if loc.content and loc.content.cook(loc, self.recipe).name == 'waste':
                            continue
                        activates.append([i,loc_name] )
                        actions.append(f"activate {loc_name}")

            struct_actions = {
                'noop': noops,
                'goto': gotos,
                'put': puts,
                'get': gets,
                'activate': activates
            }

            all_actions.append(actions)
            all_struct_actions.append(struct_actions)

        if return_struct:
            return all_struct_actions
        else:
            return all_actions
    @property
    def unwrapped(self):
        """Returns the base non-wrapped environment.

        Returns:
            Env: The base non-wrapped gym.Env instance
        """
        return self

    def reset(self, task_name=None):
        self.load_level()
        self.time_step = 0
        self.action_history = []
        self.state_history = []
        self.action_success_history = []
        self.noop_count = 0
        self.failed_action_count = 0

        self._episode_info = TaskManagerReturnType()
        self.task_manager.reset(task_name, tools_mapping=self.name_mapping)
        self.tool_cnt = defaultdict(int)

        self.success_count = 0
        self.failed_count = 0

        # self.setup_world()
        if hasattr(self, 'name_mapping'):
            for key, value in self.name_mapping.items():
                del value

        return self.all_state()

    def step(self, actions: List[str]):
        # clear feedback buffer
        self.feedback = []
        self.suggestions = []

        self.action_history.append(actions)
        self.previous_actions = actions
       # parse the LLM generated dispatching command
        valid , predicates, args, ignored_actions = self.validate_n_parse(actions)
        if predicates:
            noop_count = 0
            for predicate in predicates:
                if predicate == 'noop':
                    noop_count += 1
            if noop_count == len(predicates):
                self.feedback.append('None of the agents performed any actions which is not normal.')
        action_successes = []
        if valid:
            # executate the command

            for predicate, arg, ignored_action in zip(predicates, args, ignored_actions):
                if ignored_action:
                    action_successes.append(False)
                    continue

                agent = arg[0]
                if len(arg) == 2:
                    location = arg[1]
                    item = None
                elif len(arg) == 3:
                    item = arg[1]
                    location = arg[2]
                else:
                    location = None
                    item = None
                if location:
                    location = self.name_mapping[location]

                if item:
                    item = Item(ItemName.from_str(item))

                if predicate == 'noop':
                    action_successes.append(ActionLib.noop(self.agents[agent], self))
                    self.noop_count += 1
                elif predicate == 'goto':
                    action_successes.append(ActionLib.goto(self.agents[agent], location, self))
                elif predicate == 'activate':
                    action_successes.append(ActionLib.activate(self.agents[agent], location, self.recipe, self))
                elif predicate == 'get':
                    action_successes.append(ActionLib.get(self.agents[agent], item, location, self))
                elif predicate == 'put':
                    action_successes.append(ActionLib.put(self.agents[agent], item, location, self.recipe, self))
        else:
            action_successes = [False] * len(actions)

        for sus in action_successes:
            if not sus:
                self.failed_action_count += 1

        self.action_success_history.append(action_successes)
        # check the refresh/occupying flag of agent and location
        for agent in self.agents:
            if agent.is_occupied > 0:
                agent.is_occupied -= 1

        for loc_name, loc in self.name_mapping.items():
            if loc.is_occupied > 0:
                loc.is_occupied -= 1

        episode_info = self.done()
        # TODO(jxma): legacy, just return whether there is at least one task just success

        success = len(episode_info.task_just_success) > 0
        self._episode_info = episode_info

        self.time_step += 1
        self.task_manager.tick()

        state = self.all_state()
        self.state_history.append(state)

        reward = -0.05
        # if there is a task just success add reward by 10 and remaining_time
        if success:
            for remaining_time in episode_info.just_success_remaining_time:
                tmp = 50 +  remaining_time
                reward += tmp

            self.success_count += len(episode_info.task_just_success)

        if episode_info.just_failed:
            # reward -= 20
            self.failed_count += 1


        if np.all(action_successes):
            reward += 0.02

        reward += self.task_manager.compute_reward(self.name_mapping, self.agents)
        return state, success, {'action_success': action_successes, 'reward': reward}

    def all_state(self):
        # TODO(jxma): legacy, the first task in the current task list
        if self.task_manager.current_tasks():
            task = self.task_manager.current_tasks()[0][0]
        else:
            task = ''

        tmp = self.task_manager.current_tasks()

        agents = []
        for i in range(self.num_agents):
            agent = {}
            agent['location'] = self.agents[i].location.name
            agent['occupied'] = self.agents[i].is_occupied
            agent['id'] = i
            if self.agents[i].holding:
                agent['hold'] = self.agents[i].holding.name
                if self.agents[i].holding.name == 'waste':
                    self.suggestions.append("put the waste into the storage")
            else:
                agent['hold'] = None
            if self.agents[i].is_occupied:
                agent['occupy'] = True
            else:
                agent['occupy'] = False
            agents.append(agent)

        locations = []
        for loc_name, loc in self.name_mapping.items():
            location = {}
            location['id'] = loc_name
            if loc.content:
                location['content'] = loc.content.name
                if 'waste' in loc.content.name:
                    self.suggestions.append("put the waste into the storage")
            else:
                location['content'] = None
            if loc.is_occupied:
                location['occupy'] = True
            else:
                location['occupy'] = False

            locations.append(location)

        return_object = StepReturnType(
                        current_level=self.level,
                        current_tasks_name=[i[0] for i in tmp],
                        current_tasks_lifetime=[i[1] for i in tmp],
                        current_step=self.time_step,
                        max_steps=self.max_steps,
                        game_over=self._episode_info.game_over,
                        task_just_success=copy.deepcopy(self._episode_info.task_just_success),
                        task_just_success_location=copy.deepcopy(self._episode_info.task_just_success_location),
                        locations=locations,
                        agents=agents,
                        accomplished_tasks=self.task_manager.accomplished_tasks(),
                        just_failed=self._episode_info.just_failed)
        return return_object


class Agent:

    def __init__(self, ind: int, world:World) -> None:

        self.holding : Optional[Item] = None
        self.is_occupied: int = 0
        self.world = world
        self.location = self.world.servingtable0
        self.name = 'agent_' + str(ind)


class ActionLib:
    # all the actions
    # (args_1, args_2, ...) -> (True/False), also modify the state set if True
    @staticmethod
    def noop(agent: Agent, world: World) -> bool:
        return True

    @staticmethod
    def goto(agent: Agent, location: Location, world: World) -> bool:
        if not agent.is_occupied:
            agent.location = location
            return True
        else:
            return False

    @staticmethod
    def get(agent: Agent, item: Item, location: Location, world: World) -> bool:
        # item can be None, which means it will take whataver it is inside `location`

        # agent must be at the location
        if agent.location.name != location.name or agent.location.type != location.type:
            # print('not at location, get failed')
            world.feedback.append(f'{agent.name.replace("_", "")} is not located in {location.name}')
            return False

        # TODO(jxma): possible reason: location being serving table, location has
        # not been activated (except for storage and chopboard), etc
        if not location.pickable:
            world.feedback.append(f'{agent.name.replace("_", "")} is not allowed to pick up from {location.name} at this point')
            return False

        # agent must be unoccupied and hold nothing, location must be unoccupied
        if not agent.is_occupied and not agent.holding and not location.is_occupied:
            # if the tool is storage, just get anything as ong as it's base ingreidients that is allowed ()
            if location.type == LocationType.STORAGE:
                if item and item.name in base_ingridients:
                    agent.holding = item
                    #TODO(jxma): reward shaping awaiting refactoring
                    world.task_manager.reward_memory[agent.name] = []
                    world.task_manager.reward_memory[location.name] = []
                    return True
                else:
                    world.feedback.append(f'{agent.name.replace("_", "")} can only pickup base ingredients from the storage')
                    return False

            # Now since we only allow picking up after activation, it is impossible to pick up more than 1 items from any location or pick up from an empty tool
            # <del>TODO(jxma): we just assume everything will be obtained from the location; therefore `item`` is not used
            # if there are multiple items, we just get waste.</dev>
            if location.quantity > 1:
                assert False
                agent.holding = Item(ItemName.waste)
                location.content = None
                #TODO(jxma): reward shaping awaiting refactoring
                world.task_manager.reward_memory[agent.name] = []
                world.task_manager.reward_memory[location.name] = []
                return True
            else:
                if location.content:
                    agent.holding = location.content
                    location.content = None
                    #TODO(jxma): reward shaping awaiting refactoring
                    world.task_manager.reward_memory[agent.name] = []
                    world.task_manager.reward_memory[location.name] = []
                    location.toggle_pickable()
                    if location.type != LocationType.STORAGE and location.type != LocationType.CHOPBOARD:
                        assert location.pickable == False
                    return True
                else:
                    world.feedback.append(f'{location} is empty, you cannot get anything from there')
                    return False
        else:
            if agent.is_occupied:
                world.feedback.append(f'{agent.name.replace("_", "")} is occupied therefore cannot take any actions other than noop')
            if agent.holding:
                world.feedback.append(f'{agent.name.replace("_", "")} is holding objects, therefore cannot get objects from the tool')
            if location.is_occupied:
                world.feedback.append(f'{location.name} is occupied, therefore {agent.name.replace("_", "")} cannot get objects from the tool')
            return False

    @staticmethod
    def put(agent: Agent, item: Item, location: Location, recipe: RECIPE, world: World) -> bool:
        # agent must be at the location
        if agent.location.name != location.name or agent.location.type != location.type:
            world.feedback.append(f'{agent.name.replace("_", "")} is not located in {location.name}')
            return False

        # TODO(jxma): for serving table, you can only put dishes that need to be completed in this level
        if location.type == LocationType.SERVINGTABLE and agent.holding:
            if agent.holding.name not in world.task_manager._all_tasks:
                world.feedback.append(f'{agent.holding.name} is not any dish needed in this level and cannot be put on {location.name}')
                return False

        # TODO(jxma): for storage, you can put whatever into it whenever you want to
        # TODO(jxma): for other tools, raise error when putting irrelavent items to some tool, based off the recipe of this level
        if location.type != LocationType.SERVINGTABLE and location.type != LocationType.STORAGE and agent.holding is not None:
            allow_combo = []
            allow_item = []

            for k, v in recipe.raw_recipe.items():
                if v['location'] == location.type.value:
                    allow_combo.append(v['ingredients'])

            for combo in allow_combo:
                if location.content:
                    if all(i in combo for i in location.content.all_content()):
                        tmp = Counter(combo) - Counter(location.content.all_content())
                        allow_item.extend(list(tmp.keys()))
                else:
                    allow_item.extend(list(Counter(combo).keys()))

            # TODO(jxma): putting items that can be merged with existing content is OK
            if not agent.holding.name in allow_item:
                if not location.content or not agent.holding.name in location.content.all_content():
                    world.feedback.append(f'putting {agent.holding.name} into {location.name} will result in waste')
                    return False

        # agent must be unoccupied and hold something, location must be unoccupied and does not reach its capacity
        if not agent.is_occupied \
            and agent.holding \
                and not location.is_occupied \
                    and (location.quantity < location.capacity or \
                        location.capacity == -1 or \
                            (location.content and agent.holding.name in location.content.all_content())):
            # if the tool is a storage, just clear the agent
            # TODO(jxma): we assume everything will be put into the location; therefore `item`` is not used
            if location.type != LocationType.STORAGE:
                location.add(agent.holding)
            agent.holding = None
            #TODO(jxma): reward shaping awaiting refactoring
            world.task_manager.reward_memory[agent.name] = []
            world.task_manager.reward_memory[location.name] = []
            return True
        else:
            if agent.is_occupied:
                world.feedback.append(f'{agent.name.replace("_", "")} is occupied therefore cannot take any actions other than noop')
            if not agent.holding:
                world.feedback.append(f'{agent.name.replace("_", "")} is not holding objects, therefore cannot put objects into the tool')
            if location.is_occupied:
                world.feedback.append(f'{location.name} is occupied, therefore cannot put objects into it')
            if location.quantity >= location.capacity and location.capacity != -1:
                world.feedback.append(f'{location.name} reaches maximum capacity')
            return False

    @staticmethod
    def activate(agent: Agent, location: Location, recipe: RECIPE, world: World) -> bool:
        # chopboard and blender -- agent has to be there
        # pan, steamer, etc -- agent can do other stuffs

        # agent must be at the location
        if agent.location.name != location.name:
            world.feedback.append(f'{agent.name.replace("_", "")} is not located in {location.name}')
            return False

        # agent must be unoccupied and location must be unoccupied and has content inside (TODO: hold nothing)
        if not agent.is_occupied and not agent.holding and not location.is_occupied and location.content:

            # reject activate action that will lead to waste
            tmp = location.content.cook(location, recipe=recipe)
            if tmp.name.startswith('waste') :
                world.feedback.append(f'activating {location.name} will result in waste')
                return False
            location.content = tmp
            # set the location to be occupied and do the mixup magic
            location.is_occupied = location.refresh_time
            # if the location needs the agent to watch, set the agent to be occupied
            if location.need_watch:
                agent.is_occupied = location.refresh_time
            # TODO(jxma): reward shaping awaiting refactoring
            world.task_manager.reward_memory[location.name] = []
            # FIXME(jxma): this could be an issue as sometime you can activate a location twice without taking anything from it, ex.
            # put -> activate -> put -> activate (pasta, as in previous revision)
            location.toggle_pickable()
            assert location.pickable == True
            return True
        else:
            if agent.is_occupied:
                world.feedback.append(f'{agent.name.replace("_", "")} is occupied therefore cannot take any actions other than noop')
            if agent.holding:
                world.feedback.append(f'{agent.name.replace("_", "")} is holding objects, therefore cannot activate tools')
            if location.is_occupied:
                world.feedback.append(f'{location.name} is occupied, therefore cannot be activated')
            if not location.content:
                world.feedback.append(f'{location.name} is empty, therefore cannot be activated')
            return False
