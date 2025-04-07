import re

from torch import tensor  # noqa

from deval.tasks.task import TasksEnum


def process_line(line: str) -> tuple[str, dict]:
    pat = re.compile(r'model_url=\'(.+?)\'.*?\{(.*)\}')
    match = pat.search(line)
    if match is None:
        return '', {}

    model_url = match.group(1)
    rewards = match.group(2)
    rewards_dict = eval('{' + rewards + '}')
    return model_url, rewards_dict


model_rewards = {
    "snx999/dev2": {
        TasksEnum.RELEVANCY.value: [],
        TasksEnum.HALLUCINATION.value: [],
        TasksEnum.ATTRIBUTION.value: [],
        TasksEnum.COMPLETENESS.value: [],
    },
    "snx999/dev5": {
        TasksEnum.RELEVANCY.value: [],
        TasksEnum.HALLUCINATION.value: [],
        TasksEnum.ATTRIBUTION.value: [],
        TasksEnum.COMPLETENESS.value: [],
    },
    "vilija19/test5": {
        TasksEnum.RELEVANCY.value: [],
        TasksEnum.HALLUCINATION.value: [],
        TasksEnum.ATTRIBUTION.value: [],
        TasksEnum.COMPLETENESS.value: [],
    },
    "snx999/dev6": {
        TasksEnum.RELEVANCY.value: [],
        TasksEnum.HALLUCINATION.value: [],
        TasksEnum.ATTRIBUTION.value: [],
        TasksEnum.COMPLETENESS.value: [],
    },
}

with open('output.txt') as f:
    for line in f:
        if line.startswith('Rewards'):
            model_url, rewards_dict = process_line(line)
            if model_url:
                # print(f"Model URL: {model_url}")
                for task_name, reward in rewards_dict.items():
                    model_rewards[model_url][task_name].extend(reward)
                    # print(f"Task: {task_name}, Reward: {reward}")
            else:
                print("No valid model URL found in line.")


for model_url, rewards_dict in model_rewards.items():
    for task_name, rewards in rewards_dict.items():
        print(model_url, task_name, sep=':', end='\t')
print()
for i in range(len(model_rewards["snx999/dev2"][TasksEnum.RELEVANCY.value])):
    for model_url, rewards_dict in model_rewards.items():
        for task_name, rewards in rewards_dict.items():
            print(rewards[i].item(), end='\t')
    print()
