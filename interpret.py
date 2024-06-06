import os
import asyncio
from src.paths import get_neuron_recods_save_dir
from src.interpret import interpret_neuron_records, NeuronRecords


async def main(args):
    # get path to neuron records
    neuron_records_path = os.path.join(
        args.record_save_dir, "neuron_records_" + args.wandb_id + ".json"
    )
    print("Loading Neuron Records from: ", neuron_records_path)
    neuron_records = NeuronRecords.load(neuron_records_path)

    # interpret neuron records
    interpretation_results = await interpret_neuron_records(
        neuron_records=neuron_records,
        eval_num_features=2,
    )

    print(interpretation_results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--wandb_id",
        type=str,
        required=True,
        help="wandb run id to load the neruon records from.",
    )

    parser.add_argument(
        "--record_save_dir", type=str, default=get_neuron_recods_save_dir()
    )

    args = parser.parse_args()

    asyncio.run(main(args))
