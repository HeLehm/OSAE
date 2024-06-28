import os
import asyncio
import json
from src.paths import get_neuron_recods_save_dir, get_interpretability_save_dir
from src.interpret import interpret_neuron_records, NeuronRecords
from create_neuron_records import get_neuron_record_save_path


def get_interpretability_save_path(wandb_id):
    return os.path.join(
        get_interpretability_save_dir(), f"interpretability_{wandb_id}.json"
    )


async def interpret(args):
    neuron_records_path = get_neuron_record_save_path(args)
    print("Loading Neuron Records from: ", neuron_records_path)
    neuron_records = NeuronRecords.load(neuron_records_path)

    # interpret neuron records
    interpretation_results = await interpret_neuron_records(
        neuron_records=neuron_records,
        eval_num_features=150,
    )
    # explainer_model_name="gpt-4o",
    # explainer_prompt_format=PromptFormat.HARMONY_V4,
    # simulator_model_name="davinci-002",
    # simulator_prompt_format=PromptFormat.NONE,

    # save interpretation results
    interpretation_save_path = get_interpretability_save_path(args.wandb_id)
    print("Saving Interpretation Results to: ", interpretation_save_path)

    with open(interpretation_save_path, "w") as f:
        json.dump(interpretation_results, f, indent=4)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--wandb_id",
        type=str,
        help="wandb run id to load the neruon records from.",
    )

    parser.add_argument(
        "--record_save_dir", type=str, default=get_neuron_recods_save_dir()
    )

    args = parser.parse_args()

    asyncio.run(interpret(args))
