import pickle
import time

import bittensor as bt
from compute_horde_sdk.v1 import (
    ComputeHordeClient as SDKComputeHordeClient,
    HTTPInputVolume,
    HuggingfaceInputVolume,
    ExecutorClass,
    ComputeHordeJobStatus,
)

from deval.contest import DeValContest
from deval.model.model_state import ModelState
from deval.s3 import upload_data_to_s3
from deval.compute_horde_settings import (
    COMPUTE_HORDE_ARTIFACT_OUTPUT_PATH,
    COMPUTE_HORDE_FACILITATOR_URL,
    COMPUTE_HORDE_JOB_NAMESPACE,
    COMPUTE_HORDE_VALIDATOR_HOTKEY,
    COMPUTE_HORDE_EXECUTOR_CLASS,
    COMPUTE_HORDE_JOB_DOCKER_IMAGE,
    COMPUTE_HORDE_JOB_TIMEOUT,
    COMPUTE_HORDE_ARTIFACTS_DIR,
    COMPUTE_HORDE_VOLUME_MODEL_PATH,
    COMPUTE_HORDE_VOLUME_TASK_REPO_PATH,
)
from deval.task_repository import TaskRepository

_REQUIRED_SETTINGS = (
    "COMPUTE_HORDE_JOB_NAMESPACE",
    "COMPUTE_HORDE_JOB_DOCKER_IMAGE",
    "COMPUTE_HORDE_VALIDATOR_HOTKEY",
)


class ComputeHordeClient:
    def __init__(self, keypair: bt.Keypair):
        missing_settings = [
            setting for setting in _REQUIRED_SETTINGS if not globals().get(setting)
        ]
        if missing_settings:
            raise ValueError(
                f"Required settings: {', '.join(missing_settings)} are not set. "
                "Please set them in your .env file."
            )

        self.keypair = keypair
        self.executor_class = (
            ExecutorClass(COMPUTE_HORDE_EXECUTOR_CLASS)
            if COMPUTE_HORDE_EXECUTOR_CLASS
            else ExecutorClass.always_on__llm__a6000
        )
        self.client = SDKComputeHordeClient(
            hotkey=self.keypair,
            compute_horde_validator_hotkey=COMPUTE_HORDE_VALIDATOR_HOTKEY,
            **(
                {"facilitator_url": COMPUTE_HORDE_FACILITATOR_URL}
                if COMPUTE_HORDE_FACILITATOR_URL
                else {}
            ),
        )

    async def run_epoch_on_compute_horde(
        self,
        contest: DeValContest,
        miner_state: ModelState,
        task_repo: TaskRepository,
    ) -> ModelState:
        # TODO: Avoid using S3, if possible.
        bt.logging.info("Saving tasks to S3")
        task_repo_url = upload_data_to_s3(pickle.dumps(task_repo))
        bt.logging.info(f"Task repository uploaded to: {task_repo_url}")

        bt.logging.info("Running organic job on Compute Horde.")

        job = await self.client.create_job(
            executor_class=self.executor_class,
            job_namespace=COMPUTE_HORDE_JOB_NAMESPACE,
            docker_image=COMPUTE_HORDE_JOB_DOCKER_IMAGE,
            args=[
                "poetry",
                "run",
                "python",
                "neurons/compute_horde_entrypoint.py",
                "--uid",
                str(miner_state.uid),
                "--hf-id",
                miner_state.get_model_url(),
                "--coldkey",
                miner_state.coldkey,
                "--netuid",
                str(miner_state.netuid),
                "--forward-start-time",
                str(int(contest.start_time_datetime.timestamp())),
                "--timeout",
                str(contest.timeout),
            ],
            artifacts_dir=COMPUTE_HORDE_ARTIFACTS_DIR,
            # TODO: Replace HTTP volumes with Inline volumes, when available.
            # TODO: Pin huggingface volume revision.
            input_volumes={
                COMPUTE_HORDE_VOLUME_TASK_REPO_PATH: HTTPInputVolume(url=task_repo_url),
                COMPUTE_HORDE_VOLUME_MODEL_PATH: HuggingfaceInputVolume(
                    repo_id=miner_state.get_model_url()
                ),
            },
        )

        start = time.time()

        await job.wait(timeout=COMPUTE_HORDE_JOB_TIMEOUT)

        time_took = time.time() - start

        if job.result is None:
            raise RuntimeError("Job result is None. Check logs for more details.")

        bt.logging.info(job.result.stdout)

        if job.status != ComputeHordeJobStatus.COMPLETED:
            raise RuntimeError(
                f"Job status is {job.status}. Check logs for more details."
            )

        bt.logging.success(f"Job finished in {time_took} seconds")

        model_state_pkl = job.result.artifacts[COMPUTE_HORDE_ARTIFACT_OUTPUT_PATH]

        miner_state: ModelState = pickle.loads(model_state_pkl)
        return miner_state
