import json
from enum import Enum

# task interval and lifetime
ALPHA = 2.2
BETA = 2.5
def GAMMA(num_agents):
    # f(x) = 1/(x+1/4) + 1/5
    #return 1 / (num_agents + 1/4) + 1/5
    return 1

class LocationType(Enum):
    SERVINGTABLE = 'servingtable'
    PAN = 'pan'
    BLENDER = 'blender'
    STORAGE = 'storage'
    CHOPBOARD = 'chopboard'
    FRYER = 'fryer'
    MIXER = 'mixer'
    POT = 'pot'
    STEAMER = 'steamer'
    OVEN = 'oven'


all_actions = [ "noop", "goto", "put", "activate", "get"]
all_location_types = ["servingtable", "pan", "blender", "storage", 'chopboard', "fryer",
                      "mixer", "pot", "steamer", "oven"]

class StepReturnType:
    def __init__(self, current_level, current_tasks_name,
                 current_tasks_lifetime, current_step, max_steps,
                 game_over, task_just_success, task_just_success_location,
                 locations, agents, accomplished_tasks, just_failed) -> None:

        self.current_level = current_level
        self.current_tasks_name = current_tasks_name
        self.current_tasks_lifetime = current_tasks_lifetime
        self.current_step = current_step
        self.max_steps = max_steps
        self.game_over = game_over

        self.task_just_success = task_just_success
        self.task_just_success_location = task_just_success_location
        self.locations = locations
        self.agents = agents
        self.accomplished_tasks = accomplished_tasks
        self.just_failed = just_failed

class StrEnum():

    def __init__(self, objects) -> None:
        self.waste = 'waste'
        for object in objects:
            if not hasattr(self, object):
                setattr(self, object, object)

    def from_str(self, label:str):
        if hasattr(self, label):
           return getattr(self, label)

        return self.waste
        #raise RuntimeError(f"attribute {label} does not exist for class StrEnum")

with open("assets/recipe.json", 'r') as f:
    recipe = json.load(f)

all_objecsts = []
for key, value in recipe.items():
    all_objecsts.append(key)
    all_objecsts.extend(value['ingredients'])
all_objecsts = sorted(list(set(all_objecsts)))
ItemName = StrEnum(all_objecsts)


meat = ["pork", "beef","tuna", "salmon", "lamb", "chicken",  "turkey","egg", "duck", "lobster", "pepperoni"]
vegetables = ["potato", "carrot", "onion", 'lettuce', 'tomato', 'cucumber', 'leek', "broccoli"]
others = ['flour', 'rice', 'pasta', 'dough', "cheese", "seaweedSheet", "bread", "tortilla"]
base_ingridients = meat + vegetables + others


type_table = {
    "storage": LocationType.STORAGE,
    "blender": LocationType.BLENDER,
    "chopboard": LocationType.CHOPBOARD,
    "pan": LocationType.PAN,
    "servingtable": LocationType.SERVINGTABLE,
    "fryer": LocationType.FRYER,
    "mixer": LocationType.MIXER,
    "pot": LocationType.POT,
    "steamer": LocationType.STEAMER,
    "oven": LocationType.OVEN
}


occupied_time = {
    "storage": 0,
    "blender": 2,
    "chopboard": 1,
    "pan":2,
    "servingtable": 0,
    "fryer": 2,
    "mixer": 0,
    "pot": 3,
    "steamer": 3,
    "oven": 3
}

# -1 mean infinite
capacity = {
    "storage": -1,
    "blender": 5,
    "chopboard": 1,
    "pan": 5,
    "servingtable": -1,
    "fryer": 5,
    "mixer": 5,
    "pot": 5,
    "steamer": 5,
    "oven": 5
}

need_watch = {
    "storage": False,
    "blender": False,
    "chopboard": True,
    "pan": False,
    "servingtable": False,
    "fryer": False,
    "mixer": False,
    "pot": False,
    "steamer": False,
    "oven": False
}


example_level = ["storage", "servingtable", "blender","pan", "fryer", "mixer", "chopboard", "chopboard", ]