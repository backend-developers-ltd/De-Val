import asyncio

import bittensor as bt

from deval.model.chain_metadata import ChainModelMetadataStore
from deval.protocol import DendriteModelQueryEvent, get_metadata_from_miner

netuid = 15


async def main():
    wallet = bt.wallet(name="default", hotkey="default")

    subtensor = bt.subtensor()
    metagraph = subtensor.metagraph(netuid=netuid)

    dendrite = bt.dendrite(wallet=wallet)

    class Validator:
        def __init__(self, metagraph, dendrite):
            self.metagraph = metagraph
            self.dendrite = dendrite

    validator = Validator(metagraph, dendrite)

    for uid in range(metagraph.n.item()):
        hotkey = metagraph.hotkeys[uid]
        try:
            metadata = bt.core.extrinsics.serving.get_metadata(subtensor, netuid, hotkey)

            store = ChainModelMetadataStore(subtensor, None, netuid)

            parsed_metadata = store.parse_chain_data(metadata)

            responses = await get_metadata_from_miner(validator, uid)
            response_event = DendriteModelQueryEvent(responses)

            if parsed_metadata.model_url and response_event.repo_id and response_event.model_id:
                dendrite_model_url = f"{response_event.repo_id}/{response_event.model_id}"
                print(parsed_metadata.model_url == dendrite_model_url, parsed_metadata.model_url, dendrite_model_url)
            else:
                print('skip')
        except Exception as e:
            print('skip', e)


asyncio.run(main())
