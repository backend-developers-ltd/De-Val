import bittensor as bt

from deval.model.chain_metadata import ChainModelMetadataStore

netuid = 15

subtensor = bt.subtensor()
metagraph = subtensor.metagraph(netuid=netuid)

for hotkey in metagraph.hotkeys:
    try:
        metadata = bt.core.extrinsics.serving.get_metadata(subtensor, netuid, hotkey)

        store = ChainModelMetadataStore(subtensor, None, netuid)

        if metadata:
            parsed_metadata = store.parse_chain_data(metadata)

            if parsed_metadata.model_url:
                print(parsed_metadata)
            else:
                print('skip')
    except Exception as e:
        print(e)