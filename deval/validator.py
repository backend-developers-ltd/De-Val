import bittensor as bt
import time
from deval.base.validator import BaseValidatorNeuron
from deval.rewards.reward import RewardResult
from deval.rewards.pipeline import RewardPipeline
from deval.task_repository import TaskRepository
from dotenv import load_dotenv, find_dotenv
from deval.utils.uids import get_top_incentive_uids, get_candidate_uids
from deval.model.model_state import ModelState
from deval.model.huggingface_model import HuggingFaceModel
from deval.contest import DeValContest
from deval.protocol import get_metadata_from_miner, DendriteModelQueryEvent
from deval.agent import HumanAgent
from deval.protocol import init_request_from_task, BtEvalResponse
from deval.api.miner_docker_client import MinerDockerClient
from deval.tasks.task import Task
from deval.utils.logging import WandBLogger



class Validator(BaseValidatorNeuron):
    """
    Text prompt validator neuron.
    """

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        # load all of our environment variables for easy access
        _ = load_dotenv(find_dotenv())

        # TODO: turn to config params
        self.num_uids_total = 256
        self.max_model_size_gbs = 18 # allows for 8B models
        self.miner_incentive_threshold = self.config.neuron.miner_incentive_threshold
        self.queried_uids = set()

        # get allowed model ids and init generator
        self.allowed_models = self.config.neuron.model_ids.split(",")

        # Filter out tasks with 0 probability
        self.task_sample_rate = [
            (task, int(self.config.neuron.num_task_examples * p))
            for task, p in zip(self.config.neuron.tasks, self.config.neuron.task_p)
            if p > 0
        ]
        # Load the reward pipeline
        active_tasks = [t[0] for t in self.task_sample_rate]
        self.reward_pipeline = RewardPipeline(
            selected_tasks=active_tasks, device=self.device
        )


        self.miner_docker_client = MinerDockerClient()
        self.wandb_logger = WandBLogger(
            self.wallet.hotkey.ss58_address, 
            self.metagraph.netuid, 
            active_tasks,
            config)

        bt.logging.info("load_state()")
        self.load_state()
        self.weights = []

    # TODO: these are only staticmethods to enable early testability while bringing similar components to same place
    # remove staticmethod once we have more mature testing suite
    # because we don't pass self, we have to pass a lot of variables around
    async def forward(self):
        bt.logging.info("🚀 Starting forward loop...")
        forward_start_time = time.time()

        # init this rounds contest 
        if self.start_over:
            self.contest = DeValContest(
                self.reward_pipeline, 
                forward_start_time, 
                self.config.neuron.timeout
            )
            self.task_repo = TaskRepository(allowed_models=self.allowed_models)
        
            # collect the top incentive uids
            top_incentive_uids = get_top_incentive_uids(self, k=self.miner_incentive_threshold, num_uids=self.num_uids_total).to(self.device)
            available_uids = get_candidate_uids(self, k = self.num_uids_total)

            # generate all tasks for miners to be evaluated on
            self.task_repo.generate_all_tasks(task_probabilities=self.task_sample_rate)

            

            # reset complete
            self.start_over = False
        else:
            available_uids = get_candidate_uids(self, k = self.num_uids_total)
            available_uids = [uid for uid in available_uids if uid not in self.queried_uids]

        for uid in available_uids:
            try:
                # get the model metadata information from miner
                responses = await get_metadata_from_miner(self, uid)
                response_event = DendriteModelQueryEvent(responses)
                bt.logging.info(f"Created DendriteResponseEvent:\n {response_event}") 

                miner_state = ModelState(response_event.repo_id, response_event.model_id, uid)

                is_valid = miner_state.should_run_evaluation(
                    uid, self.max_model_size_gbs, forward_start_time, top_incentive_uids
                )

                if is_valid:
                    miner_state = Validator.run_epoch(
                        self.contest,
                        miner_state, 
                        self.task_repo, 
                        self.miner_docker_client,
                        self.wandb_logger
                    )

                # update contest
                self.contest.update_model_state_with_rewards(miner_state) 
                self.queried_uids.add(uid)
                self.save_state()
            except Exception as e:
                self.queried_uids.add(uid)
                bt.logging.info(f"Error in forward pass for uid: {uid} skipping to next round. Exception: {e}")

        
        self.weights = self.contest.rank_and_select_winners(self.task_sample_rate)
        self.set_weights()
        self.start_over = True
        self.reset()

        
    @staticmethod
    def run_epoch(
        contest: DeValContest, 
        miner_state: ModelState, 
        task_repo: TaskRepository, 
        miner_docker_client: MinerDockerClient,
        wandb_logger: WandBLogger,
    ):
        #pull model, update contest, and validate model 
        model_dir = HuggingFaceModel.pull_model_and_files(miner_state.get_model_url())
        is_valid = contest.validate_model(miner_state, model_dir)
        if not is_valid:
            return miner_state

        valid_connection = miner_docker_client.initialize_miner_api()

        # run through all tasks if we can connect, otherwise skip
        if valid_connection:
            for task_name, tasks in task_repo.get_all_tasks():
                miner_state = Validator.run_step(
                    task_name, 
                    tasks, 
                    miner_docker_client, 
                    miner_state, 
                    contest, 
                    wandb_logger
                )


        # cleanup and remove the model 
        miner_state.cleanup(model_dir)
        
        return miner_state

    @staticmethod
    def run_step(
        task_name: str,
        tasks: list[Task], 
        docker_client: MinerDockerClient,
        miner_state: ModelState,
        contest: DeValContest,
        wandb_logger: WandBLogger
    ):
        
        responses = []

        for task in tasks:
            # query docker container with task 
            agent = HumanAgent(
                task=task
            )
            request = init_request_from_task(task)
            response = docker_client.query_eval(request, contest.timeout)
            bt_response = BtEvalResponse(
                uid = miner_state.uid,
                response = response,
                human_agent = agent
            )

            responses.append(bt_response)

            
            
        # generate and store reward  
        reward_result = RewardResult(
            contest.reward_pipeline,
            responses=responses,
            device="cpu" # self.device,
        )
        wandb_logger.log_event(responses, reward_result)
        
        miner_state.add_reward(task_name, reward_result)


        return miner_state




    def __enter__(self):
        if self.config.no_background_thread:
            bt.logging.warning("Running validator in main thread.")
            self.run()
        else:
            self.run_in_background_thread()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Stops the validator's background operations upon exiting the context.
        This method facilitates the use of the validator in a 'with' statement.

        Args:
            exc_type: The type of the exception that caused the context to be exited.
                      None if the context was exited without an exception.
            exc_value: The instance of the exception that caused the context to be exited.
                       None if the context was exited without an exception.
            traceback: A traceback object encoding the stack trace.
                       None if the context was exited without an exception.
        """
        if self.is_running:
            bt.logging.debug("Stopping validator in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")
