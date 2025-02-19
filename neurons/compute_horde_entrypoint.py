import argparse
import os
import pickle
import subprocess

import bittensor as bt
import torch

from deval.api.miner_docker_client import MinerDockerClient
from deval.contest import DeValContest
from deval.model.model_state import ModelState
from deval.rewards.pipeline import RewardPipeline
from deval.compute_horde_settings import (
    COMPUTE_HORDE_ARTIFACT_OUTPUT_PATH,
    COMPUTE_HORDE_VOLUME_MODEL_PATH,
    COMPUTE_HORDE_VOLUME_TASK_REPO_PATH,
)
from deval.task_repository import TaskRepository
from deval.utils.logging import WandBLogger
from deval.validator import Validator


def get_args():
    parser = argparse.ArgumentParser(description="Parse required flags.")
    parser.add_argument(
        "--hf-id", type=str, required=True, help="HF ID (string, required)"
    )
    parser.add_argument(
        "--coldkey", type=str, required=True, help="Coldkey (string, required)"
    )
    parser.add_argument("--uid", type=int, required=True, help="UID (int, required)")
    parser.add_argument(
        "--netuid", type=int, required=False, help="NetUID (int, optional)", default=15
    )
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        help="Device (string, optional)",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--forward-start-time",
        type=int,
        required=False,
        help="Forward start time (int, required)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        required=False,
        help="Timeout (int, optional)",
        default=60 * 10,
    )
    return parser.parse_args()


def main():
    args = get_args()

    bt.logging.info("Loading task repository")
    with open(COMPUTE_HORDE_VOLUME_TASK_REPO_PATH, "rb") as f:
        task_repo: TaskRepository = pickle.load(f)

    bt.logging.info("Starting miner API")
    with subprocess.Popen(
        ["poetry", "run", "uvicorn", "deval.api.miner_api:app", "--port", "8000"],
        env={
            **os.environ.copy(),
            "MODEL_VOLUME_DIR": COMPUTE_HORDE_VOLUME_MODEL_PATH,
            "PYTHONUNBUFFERED": "1",
        },
    ) as uvicorn_process:

        active_tasks = [task_name for task_name, _ in task_repo.get_all_tasks()]

        reward_pipeline = RewardPipeline(
            selected_tasks=active_tasks, device=args.device
        )

        contest = DeValContest(
            reward_pipeline=reward_pipeline,
            forward_start_time=args.forward_start_time,
            timeout=args.timeout,
        )

        repo_id, model_id = args.hf_id.split("/")
        miner_state = ModelState(repo_id, model_id, args.uid, netuid=args.netuid)
        miner_state.add_miner_coldkey(args.coldkey)

        miner_docker_client = MinerDockerClient(api_url="http://localhost:8000")

        miner_docker_client._poll_service_for_readiness(500)

        wandb_logger = WandBLogger(None, None, active_tasks, None, force_off=True)

        for task_name, tasks in task_repo.get_all_tasks():
            bt.logging.debug(f"Running task {task_name}")
            miner_state = Validator.run_step(
                task_name,
                tasks,
                miner_docker_client,
                miner_state,
                contest,
                wandb_logger,
            )

        bt.logging.debug(miner_state.rewards)
        print("Completed epoch")

        with open(COMPUTE_HORDE_ARTIFACT_OUTPUT_PATH, "wb") as f:
            pickle.dump(miner_state, f)

        print("Terminating miner API")
        uvicorn_process.terminate()


if __name__ == "__main__":
    main()
