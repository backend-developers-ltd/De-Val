# The MIT License (MIT)
# Copyright © 2024 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import time
import bittensor as bt
import argparse

# Bittensor Miner Template:
from deval.protocol import ModelQuerySynapse

# import base miner class which takes care of most of the boilerplate
from deval.base.eval_miner import Miner


class ContestMiner(Miner):

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        super().add_args(parser)

    def __init__(self, config=None):
        super().__init__(config=config)


    async def forward(self, synapse: ModelQuerySynapse) -> ModelQuerySynapse:
        """
        Processes the incoming synapse by performing a predefined operation on the input data.
        This method should be replaced with actual logic relevant to the miner's purpose.

        Args:
            synapse (ModelQuerySynapse): The synapse object containing data to point to HF repo.

        Returns:
            ModelQuerySynapse: The synapse object with complete outputs 
        """
        
        synapse.repo_id = "deval-core"
        synapse.model_id = "base-eval"

        return synapse
            
        


# This is the main function, which runs the miner.
if __name__ == "__main__":
    with ContestMiner() as miner:
        while True:
            miner.log_status()
            time.sleep(5)

            if miner.should_exit:
                bt.logging.warning("Ending miner...")
                break