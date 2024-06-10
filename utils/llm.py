import argparse
import json
import os
import random
import time
import requests
import openai
from anthropic import AI_PROMPT, HUMAN_PROMPT, Anthropic
from vertexai.preview.language_models import (ChatModel, InputOutputTextPair,
                                              TextGenerationModel)

from levels.constants import all_objecsts, base_ingridients, capacity
from levels.utils import compute_dependency
from overcooked.game import World
from openai import OpenAI
client = OpenAI()

CLAUDE_KEY = ""

key_id = 0
def rules(env, notes=True):
    prompt = ''
    prompt += 'The available actions are :\n'
    prompt += '1) goto: goto a tool location \n'
    prompt += '2) get: get some object from a tool \n'
    prompt += '3) put: put some object into a tool \n'
    prompt += '4) activate: activate the tool to cook all ingredients inside the tool into a different tools \n'
    prompt += '5) noop: not performing any actions \n'

    prompt += 'Sometimes the system will give you error messages. Please consider these error messages when executing actions.  \n'
    prompt += 'You need to specify action for all of the agents, **except human**. They all have different agent numbers. Do not assign actions to the same agent more than once. \n'
    prompt += '\n'

    if notes:
        prompt += 'When the tools reach its capacity, you need to take stuff out. Otherwise, you cannot put items inside.\n'
        prompt += 'When you are holding objects, you cannot get any more objects. \n'
        prompt += "When you are holding objects, you cannot activate tools. \n"
        prompt += "Afer you cooked a required dish, you need to put it into the servingtable. \n"
        prompt += 'You can only pick up objects from the tool location, if you are located at the tool location. \n'
        prompt += "When you activate any tools, make sure all the items inside the tool are respecting the recipes. Otherwise, you will cook waste. Avoid waste at all cost. \n"

        prompt += "*** You should mix salad in the mixer. To make salad you should chop veggies first. *** \n"
        prompt += "*** If the tool is occupied, indicated by the occupy() predicate, you cannot get objects from it or put objects into it. *** \n"
        prompt += "*** The food orders are keep coming. You should finish as many dishes as possible and finish every dish as soon as possible. Please deliver the order to the serveringtable when it is finished. *** \n"
        prompt += "*** The dish will expire after the lifetime reaches 0 and it's not at the serveringtable. Please avoid this. *** "

    prompt += 'Here are the recipes: \n'
    prompt += recipes(env)

    prompt += generate_tool_descriptions(env)
    prompt += '\n\n'

    return prompt

def recipes(env: World):
    import copy
    tasks =  copy.deepcopy(env.task_manager._all_tasks)
    tasks.append('porkMeatcake')
    required_components = []

    for task in tasks:
        required_components.extend(compute_dependency(task)[0])
    required_components = set(required_components)

    prompt = '\n'
    with open('assets/recipe.json', 'r') as f:
        recipe = json.load(f)

    task_related_objects = set()
    task_related_tools = set()
    for dish, value in recipe.items():
        if dish in required_components:
            ingredients = value['ingredients']
            location = value['location']
            prompt += f'Cook {dish} at: \n'
            prompt += f' -- location: {location} \n'
            prompt += f' -- with ingredients: '
            task_related_objects.add(dish)
            task_related_tools.add(location)
            for ingredient in ingredients:
                task_related_objects.add(ingredient)
                prompt += f'    {ingredient}, '
            prompt += '\n'

    prompt += 'The following objects are available: \n'
    for idx, item in enumerate(task_related_objects):
        prompt += f' --{idx+1}) {item} \n'
    prompt += "The objecsts are cooked using tools or are just base ingredients. \n"

    prompt += "Among them, the following are base ingredients: \n"
    cnt = 1
    for idx, item in enumerate(task_related_objects):
        if item in base_ingridients:
            prompt += f" --{cnt}) {item} \n"
            cnt += 1
    prompt += "You can only obtain base ingredients from the storage initially.  \n"

    prompt += 'Additional rules: \n'
    for tool_name, tool in env.name_mapping.items():
        cap = capacity[tool_name[:-1]]
        num = cap
        if num == -1:
            num = 'infinite'

        prompt += f'You can place up to {num} item into the {tool_name} \n'
        prompt += f'You can place up to {num} item into the {tool_name} \n'

    return prompt

def generate_tool_descriptions(env: World):
    prompt = '** Only ** the following tools are available: \n'
    for tool_name, tool in env.name_mapping.items():
        prompt += f'{tool_name}, '

    prompt += 'You cannot pick up these tools. You can only use those tools at the corresponding location.'

    prompt += '\n'
    return prompt


def prepend_prompt(prompt, add, verbose=True):
    if verbose:
        print(add)
    return prompt + add

def prepend_history(history, add, role='user', verbose=True):
    if verbose:
        print(f'\n\n[[{role}]]\n\n' + add)
    assert role in ['user', 'assistant']
    history.append((role, add))
    return history

def next_key():
    # global key_id
    # with open('./key.txt', 'r') as f:
    #     all_keys = f.read().split('\n')
    # # all_keys = open('./key.txt', 'r').read().split('\n')
    # num = len(all_keys)
    # key_id += 1

    # if key_id >= num:
    #     key_id -= num
    openai_key = os.getenv('OPENAI_KEY')
    openai.api_key = openai_key
    


def completion_llm(prompt, temperature=0, max_tokens=100):
    if openai.api_key is None:
        next_key()
    total_trials = 0
    while True:
        try:
            response =  client.chat.completions.create(
                model='text-davinci-003',
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens)
            break
        except openai.OpenAIError:
            next_key()
            total_trials += 1
    return response.choices[0].text

def chat_llm_vicuna(history, temperature=0, max_tokens=100):
    openai.api_key = "EMPTY" # Not support yet
    openai.api_base = "http://localhost:8000/v1"

    if type(history) == str:
        history = [('user', history)]

    chat_history = []
    for i in history:
        if i[0] == 'user':
            chat_history.append({
                'role': 'user',
                'content': i[1]
            })
        elif i[0] == 'assistant':
            chat_history.append({
                'role': 'assistant',
                'content': i[1]
            })
        else:
            raise NotImplementedError

    response = openai.ChatCompletion.create(
                model="vicuna-33b-v1.3",
                messages=chat_history,
                temperature=temperature,
                max_tokens=max_tokens
            )

    return response.choices[0].message.content

def chat_azure(history, temperature, max_tokens):
    url = 'https://gcrgpt4aoai4c.openai.azure.com/openai/deployments/gpt-4/chat/completions?api-version=2023-06-01-preview'
    api_key = ''
    headers = {'Content-Type': 'application/json', 'api-key': api_key}  
    chat_history = []
    for i in history:
        if i[0] == 'user':
            chat_history.append({
                'role': 'user',
                'content': i[1]
            })
        elif i[0] == 'assistant':
            chat_history.append({
                'role': 'assistant',
                'content': i[1]
            })

    data = {
        "messages":chat_history,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "n":1
    }
    

    while True:
        try:
            response = requests.post(url, json=data, headers=headers)
            return response.json()['choices'][0]['message']['content']
        except:
            pass

def chat_llm(history, temperature=0, max_tokens=100, model='gpt-4', context=''):
    if context:
        if model != 'palm-2':
            history = [('user', context)] + history
    if model =='gpt-4-azure':
        return chat_azure(history=history, temperature=temperature, max_tokens=max_tokens)
    
    if model == 'palm-2':
        return chat_palm(history, temperature, max_tokens, context)

    if model == 'claude-2':
        return chat_claude(history, temperature, max_tokens=200)
    if openai.api_key is None:
        next_key()

    if type(history) == str:
        history = [('user', history)]

    chat_history = []
    for i in history:
        if i[0] == 'user':
            chat_history.append({
                'role': 'user',
                'content': i[1]
            })
        elif i[0] == 'assistant':
            chat_history.append({
                'role': 'assistant',
                'content': i[1]
            })
        else:
            raise NotImplementedError

   

    while True:

        try:
            response = client.chat.completions.create(
                model = model,
                messages=chat_history,
                temperature=temperature,
                max_tokens=max_tokens
            )
            break
        except openai.OpenAIError as e:
            print(e)
            next_key()
            time.sleep(0.1)
    return response.choices[0].message.content

def completion_claude(prompt, temperature=0, max_tokens=100):
    anthropic = Anthropic(api_key=CLAUDE_KEY)
    try:
        completion = anthropic.completions.create(
            model="claude-2",
            max_tokens_to_sample=max_tokens,
            prompt=f'{prompt} {AI_PROMPT}',
        )
    except Exception as e:
        print(e)
        return
    return completion.completion

def chat_claude(history, temperature=0, max_tokens=100):
    if type(history) == str:
        chat_history = f"{HUMAN_PROMPT} {history}"
    else:
        chat_history = ''
        for i in history:
            if i[0] == 'user':
                chat_history += f'{HUMAN_PROMPT} {i[1]}'
            elif i[0] == 'assistant':
                chat_history += f'{AI_PROMPT} {i[1]}'
            else:
                raise NotImplementedError

    try:
        anthropic = Anthropic(api_key=CLAUDE_KEY)
        completion = anthropic.completions.create(
            model="claude-2",
            max_tokens_to_sample=max_tokens,
            prompt=f'{chat_history} {AI_PROMPT}',
        )
    except Exception as e:
        print(e)
        return
    return completion.completion.lstrip()

def completion_palm(prompt, temperature=0, max_tokens=100):
    model = TextGenerationModel.from_pretrained("text-bison@001")
    parameters = {
        "temperature": temperature,  # Temperature controls the degree of randomness in token selection.
        "max_output_tokens": max_tokens,  # Token limit determines the maximum amount of text output.
        # "top_p": 0.95,  # Tokens are selected from most probable to least until the sum of their probabilities equals the top_p value.
        # "top_k": 40,  # A top_k of 1 means the selected token is the most probable among all tokens.
    }
    try:
        response = model.predict(
            prompt,
            **parameters,
        )
    except Exception as e:
        print(e)
        return
    return response.text


def chat_palm(history, temperature=0, max_tokens=100, context=''):
    chat_model = ChatModel.from_pretrained("chat-bison@001")
    parameters = {
        "temperature": temperature,  # Temperature controls the degree of randomness in token selection.
        "max_output_tokens": max_tokens,  # Token limit determines the maximum amount of text output.
        # "top_p": 0.95,  # Tokens are selected from most probable to least until the sum of their probabilities equals the top_p value.
        # "top_k": 40,  # A top_k of 1 means the selected token is the most probable among all tokens.
    }
    if type(history) == str:
        examples = []
        prompt = history
    else:
        examples = []
        # TODO(jxma): we assume we can cumulatively aggregate user and assistant chats
        user_buffer = ''
        assistant_buffer = ''
        for i in history:
            if i[0] == 'user':
                # completing this round of chat
                if assistant_buffer != '':
                    examples.append(InputOutputTextPair(
                        input_text=user_buffer,
                        output_text=assistant_buffer
                    ))
                    user_buffer = assistant_buffer = ''
                user_buffer += (i[1] + '\n')
            elif i[0] == 'assistant':
                assistant_buffer += (i[1] + '\n')
            else:
                raise NotImplementedError
        # TODO(jxma): assume history always ends with (one or more) user messages
        assert history[-1][0] == 'user'
        prompt = user_buffer
    try:
        chat = chat_model.start_chat(
            context=context,
            examples=examples,
        )
        response = chat.send_message(prompt, **parameters)
    except Exception as e:
        print(e)
        return
    return response.text


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser(description='')
    args_parser.add_argument('--mode', type=str, choices=['completion', 'chat'])
    args_parser.add_argument('--max', type=int, default=1000)
    args = args_parser.parse_args()

    # Load your API key from an environment variable or secret management service
    openai.api_key = open('./key.txt', 'r').read()

    if args.mode == 'completion':
        query_llm = completion_llm
    else:
        query_llm = chat_llm

    prompt = open('./assets/prompt_1.txt', 'r').read()
    ret = query_llm(prompt + '\n Goal: make a meatcake with beef and flour',
                    max_tokens=args.max)
    print(ret)
