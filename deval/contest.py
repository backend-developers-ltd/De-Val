import torch
from deval.tasks.task import TasksEnum
from deval.model.model_state import ModelState
from pydantic import BaseModel
from datetime import datetime
from deval.rewards import RewardPipeline

class DeValContest:

    def __init__(self, reward_pipeline: RewardPipeline, forward_start_time: int, timeout: int):
        self.model_rewards: dict[int, dict[str, list[float]]] = {} # int = uid, str = task name, list[float] = list of rewards
        self.ranked_rewards: list[tuple(int, float)] = [] # int = uid, float = reward
        self.model_hashes: dict[str, ModelState] = {} 
        self.start_time_datetime: datetime = datetime.fromtimestamp(forward_start_time)
        self.reward_pipeline: RewardPipeline = reward_pipeline
        self.timeout: int = timeout

    def validate_model(self, miner_state: ModelState) -> bool:
        # ensure the last commit date is before forward start time
        if self.start_time_datetime < miner_state.get_last_commit_date:
            return False
        
        # compute the safetensors hash and check if duplicate. Zero out the duplicate based on last safetensors file update
        miner_state.compute_model_hash()
        duplicated_model = self.model_hashes.get(miner_state.model_hash, None)

        if duplicated_model is None:
            self.model_hashes[miner_state.model_hash] = miner_state
            return True

        else:
            # Start here
            # check which one has the most recent date
            # if current one then that is the duplicate and we return that it is an invalid model
            # if previous one then we zero out the previous model's scores and return this as a valid model
            duplicated_model_uid = duplicated_model.uid
            duplicated_model_date = duplicated_model.last_safetensor_update

            if miner_state.last_safetensor_update > duplicated_model_date:
                return False

            else: 
                # if the current model is actually the real one, then we need to update
                # the model that the hash points to and zero out the duplicate rewards 
                # update the model associated 
                self.model_hashes[miner_state.model_hash] = miner_state
                self.model_rewards.pop(duplicated_model_uid, None)
                return True

                
        
    def update_model_state_with_rewards(self, miner_state: ModelState) -> None:
        self.model_rewards[miner_state.uid] = miner_state.rewards 

    def rank(self, available_uids: list[int]):
        # takes all the models, ranks them, selects winners
        # TODO: I may need to zero out all uid rewards who were not evaluated 
        # because we need to drop their incentive 
        pass