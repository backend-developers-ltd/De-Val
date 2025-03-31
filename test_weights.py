import os
import pickle
import time
from unittest.mock import MagicMock

import bittensor
import torch
from dotenv import load_dotenv

from deval.model.model_state import ModelState

load_dotenv()

from deval.contest import DeValContest
from deval.rewards.pipeline import RewardPipeline
from deval.task_repository import TASKS, TaskRepository
from deval.tasks.task import TasksEnum
from deval.validator import Validator

bittensor.logging.set_console()

allowed_models = ["gpt-4o-mini"]

task_sample_rate = [
    (TasksEnum.RELEVANCY.value, 1),
    #(TasksEnum.HALLUCINATION.value, 1),
    #(TasksEnum.ATTRIBUTION.value, 1),
    #(TasksEnum.COMPLETENESS.value, 1)
]

# print("Initializing tasks and contest")
# if os.path.exists("task_repo.pkl"):
#     print("Loading task repo from file")
#     with open("task_repo.pkl", "rb") as f:
#         task_repo = pickle.load(f)
# else:
#     task_repo = TaskRepository(allowed_models=allowed_models)
#
#     print("Generating the tasks")
#     task_repo.generate_all_tasks(task_probabilities=task_sample_rate)
#     with open("task_repo.pkl", "wb") as f:
#         pickle.dump(task_repo, f)
# #
# active_tasks = [t[0] for t in task_sample_rate]
# reward_pipeline = RewardPipeline(
#     selected_tasks=active_tasks, device="cpu"
# )

forward_start_time = time.time()
timeout = 600
contest = DeValContest(
    None,
    forward_start_time,
    timeout
)

num_tasks = 10

for uid in range(10):
    miner_state = MagicMock(uid=uid, rewards={task_name: [] for task_name in TASKS})

    for task_name in TASKS:
        for _ in range(num_tasks):
            rwrd = torch.rand(1)
            ModelState.add_reward(miner_state, task_name, MagicMock(rewards=[rwrd]))

    contest.update_model_state_with_rewards(miner_state)


vali = MagicMock()
vali.device = "cpu"
vali.metagraph = MagicMock(n=256)
vali.scores = torch.zeros(
    vali.metagraph.n, dtype=torch.float32, device=vali.device
)

denom = len(TASKS) * num_tasks

print(contest.model_rewards)

formatted_scores = Validator.update_scores(vali, contest.model_rewards, denom)

# print(formatted_scores)

vali.weights = contest.rank_and_select_winners(formatted_scores)

print(vali.weights)
