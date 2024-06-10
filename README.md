MindAgent
===

A minimal text-based Overcooked! 2 game.

## Installation

```bash
pip install -r requirements.txt
```

## Run LLM experiments
First, need to specify openai key on the command line
```bash
export OPENAI_API_KEY='xxx'
```
for example with 2 agents, with gpt-4 and level_0
Please read agents/chat_agent.py  for more options.
```bash
python agents/chat_agent.py  --level level_0 gpt-4 --num_agents 2
```

## Run RL experiment
```bash
python train/train.py --level level_0
```

## How to collect few-shot examples using web app

1. Launch the web-app and start to collect demonstrations

    ```bash
    python webpage/app.py
    ```

2. Locate the the game in `logs/` based off `user-id`

3. Replay the log

    ```bash
    python agents/replay_amt.py --gameplay <game_log> | tee log_amt.txt
    ```

4. Convert the log into few-shot prompt example. You may specify the number of agents used used in this log

    ```bash
    python assets/log_to_prompt.py --log-input log_amt.txt --prompt-output assets/amt_examples/prompt_2agent.txt --num-agents 2
    ```

### PaLM 2

1. Sign up [here](https://cloud.google.com/vertex-ai) for Vertex AI  and get $300 credit for free

2. [Install](https://cloud.google.com/sdk/docs/install) google cloud sdk

    Remember to run `gcloud init` before anything

3. Run the following

    ```bash
    gcloud auth application-default login
    pip install pandas google-cloud-aiplatform[pipelines]==1.26.0 google-auth==2.17.3
    ```

4. You are good to go! Take a look at `completion_palm` and `chat_palm` in `llm.py`.


Put your openai API key to key.txt

## Run & Test env

```bash
python test/test_1.py
```

## Add new recipe

`assets/recipe.json`

## Add new level
`assets/tasks_level_final.json`

## unit test
```bash
pytest test
```
## unit test with stdout
```bash
pytest test -rP
```

## using coverage
```bash
coverage run -m pytest test
coverage report
coverage html
```

## webserver
```
python webpage/app.py --debug
```


## Minecraft Experiment
please refer to [Minecraft Experiemnt](https://github.com/nikepupu/Voyager)

