import argparse
import re

# first run the following
# $ python -u chat_agent.py "claude-2" | tee log.txt

def main(args):
    f = open(args.log_input, 'r').read()
    patterns = ['\[True,\s+False\]', '\[True,\s+True\]', '\[False,\s+False\]', '\[False,\s+True\]']

    # Replace specified patterns with an empty string
    for pattern in patterns:
        f = re.sub(pattern, '', f)
    d = re.split('\[\[.+\]\]', re.sub('\n{3,}', '\n\n', f))
    # remove the first line if it only contains linebreaks
    if len(set(set(d[0]))) == 1:
        d = d[1:]

    of = open(args.prompt_output, 'w')
    prompt = f"""
multiple goal for chat-based agent
###
There are {args.num_agents} agents available, so you can execute {args.num_agents} actions at a time.
This is an example you can use as a reference for a different level.
    """
    # TODO(jxma): we should try to remove some redundent linebreaks from the prompt, bottomline is to make it similar to prompt_2.txt otherwise model like PaLM-2 will break
    # make sure the demonstration end with assistant response
    if '-action:' in d[-1]:
        d = d[:-1]
    for i in d:
        # prompt += '\n'
        prompt += i[1:-2]
        prompt += '\n***'
    prompt = re.sub('\n{3,}', '\n\n', prompt)
    of.write(prompt)
    of.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert game log to few-shot examples")

    # Add the arguments
    parser.add_argument('--log-input', type=str, help='input game log file to use')
    parser.add_argument('--prompt-output', type=str, help='output few-shot prompt to produce')
    parser.add_argument('--num-agents', type=int, help='number of agents')
    # Parse the arguments
    args = parser.parse_args()

    main(args)
