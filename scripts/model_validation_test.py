import sys
import time
from dataclasses import dataclass

import bittensor as bt
from dotenv import load_dotenv, find_dotenv

from deval.contest import DeValContest
from deval.validator import Validator
from deval.rewards.pipeline import RewardPipeline
from deval.task_repository import TaskRepository
from deval.model.model_state import ModelState
from deval.tasks.task import TasksEnum
from deval.api.miner_docker_client import MinerDockerClient
from deval.utils.logging import WandBLogger
from deval.model.chain_metadata import ChainModelMetadataStore

# initialize
load_dotenv(find_dotenv())
allowed_models = ["gpt-4o", "gpt-4o-mini", "mistral-7b", "claude-3.5", "command-r-plus"]

timeout = 30 * 60
netuid = 15
max_model_size_gbs = 18

@dataclass
class Model:
    uid: int
    model_url: str
    hotkey: str
    coldkey: str

# 79, 43, 252, 175
# three good, one average
models = [
    Model(79, "snx999/dev2", "5FCV5RpK28GwTDqukKxwepzV2YuKW3J9AgMz8DFd9VKJAecw", "5HSwsfL3J3zdgYF3a3b15QSSFsgEgLTg1wg84rbweSUFjf5c"),
    Model(43, "snx999/dev5",     "5DMJyJVAXgb8j4wxjr3ZcMcgEj1Gpi27ZXqqNzKxF8oRFwNh",        "5HSwsfL3J3zdgYF3a3b15QSSFsgEgLTg1wg84rbweSUFjf5c"),
    Model(252, "vilija19/test5",  "5Di7RD3QR4Hi4PibNXQS3pDant2zxkyx2d3H65ZKVs66TaT6",        "5FX56YqiXbiK49XEwWHaqwqzEvv6zix7AiX2SqHV8aR4mjo1"),
    Model(175, "snx999/dev6",     "5FNkC4ZeZfnadUALqWhDN83yY4hfFBHDiKKN97GVCpSw9qPF",        "5HSwsfL3J3zdgYF3a3b15QSSFsgEgLTg1wg84rbweSUFjf5c"),
]

subtensor = bt.subtensor(network='finney')


print("Initializing tasks and contest")
task_repo = TaskRepository(allowed_models=allowed_models)

task_sample_rate = [
    (TasksEnum.RELEVANCY.value, 30),
    (TasksEnum.HALLUCINATION.value, 30),
    (TasksEnum.ATTRIBUTION.value, 30),
    (TasksEnum.COMPLETENESS.value, 30)
]
active_tasks = [t[0] for t in task_sample_rate]
reward_pipeline = RewardPipeline(
    selected_tasks=active_tasks, device="cuda"
)


forward_start_time = time.time()
contest = DeValContest(
    reward_pipeline,
    forward_start_time,
    timeout
)

miner_docker_client = MinerDockerClient()
wandb_logger = WandBLogger(None, None, active_tasks, None, force_off=True)
metadata_store = ChainModelMetadataStore(
    subtensor=subtensor, wallet=None, subnet_uid=netuid
)

print("Generating the tasks")
task_repo.generate_all_tasks(task_probabilities=task_sample_rate)


for model in models:
    chain_metadata = metadata_store.retrieve_model_metadata(model.hotkey)
    repo_id, model_id = model.model_url.split("/")
    miner_state = ModelState(repo_id, model_id, model.uid, netuid)
    miner_state.add_miner_coldkey(model.coldkey)
    miner_state.add_chain_metadata(chain_metadata)

    miner_state = Validator.run_epoch(
        contest,
        miner_state,
        task_repo,
        miner_docker_client,
        wandb_logger
    )
    print("Completed epoch")

    print(f"Rewards for {model=}: {miner_state.rewards}")

    print("updating contest with rewards and ranking")
    contest.update_model_state_with_rewards(miner_state)
weights = contest.rank_and_select_winners(task_sample_rate)


print(weights)
