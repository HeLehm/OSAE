import os
import wandb
import argparse

from src.interpret import generate_neuron_records
from src.paths import get_neuron_recods_save_dir, get_embeddings_cache_dir
from train_sae import get_ds, get_sae_checkpoint_path
from src.sae import load_sae


def get_wandb_run(run_id, entity, project):
    api = wandb.Api()
    return api.run(f"{entity}/{project}/{run_id}")

def get_wandb_config(run_id, entity, project):
    run = get_wandb_run(run_id, entity, project)
    return run.config

def infer_layer_idx(config):
    # try to create a better layer_index int for openai
    layer_idx = -1
    try:
        layer_idx = int(config.layername.split(".")[-1])
        print(
            f"Successfully parsed layer index ({layer_idx}) from layername ({config.layername})."
        )
    except Exception as _:
        pass

    return layer_idx

def get_neuron_record_save_path(config):
    return os.path.join(
        config.record_save_dir, f"neuron_records_{config.wandb_id}.json"
    )    


def create_neuron_records(
        config,
        recreate=False,
):
    record_save_path = get_neuron_record_save_path(config)
    if os.path.exists(record_save_path) and not recreate:
        print(f"Neuron records already exist at {record_save_path}. Skipping.")
        return record_save_path
    
    layer_idx = infer_layer_idx(config=config)

    # get dataset
    activation_dataset = get_ds(config, flatten_sequence=False)

    if hasattr(config, "debug") and config.debug:
        activation_dataset = activation_dataset[:1000]

    # load sae
    sae_ckpt_path = get_sae_checkpoint_path(config)
    print(f"Loading SAE from {sae_ckpt_path}")
    sae = load_sae(sae_ckpt_path)
    print(f"SAE {sae} loaded")

    # generate neuron records
    neuron_records = generate_neuron_records(
        activation_ds=activation_dataset,
        sae=sae,
        top_k=config.top_k,
        layer_index=layer_idx,
    )

    # save neuron records
    neuron_records.save(record_save_path)

    return record_save_path


def get_argparser():
    parser = argparse.ArgumentParser()

    # to test:
    # rg3ucesp (vanilla)
    # rmp0v3as (orthogonal)
    # rdbqr2jr orthogonal shear
    parser.add_argument(
        "--wandb_id", type=str, required=True, help="wandb run id to load the sae from."
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="bschergen",
        help="wandb entity to load the sae from.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="NNOrthogonalFinal",
        help="wandb project to load the sae from.",
    )

    parser.add_argument("--max_seq_length", type=int, default=64)
    parser.add_argument("--cache_dir", type=str, default=get_embeddings_cache_dir())
    parser.add_argument(
        "--record_save_dir", type=str, default=get_neuron_recods_save_dir()
    )

    parser.add_argument("--top_k", type=int, default=50)  # or 20?

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode. Only use a small subset of the data.",
    )
    parser.add_argument(
        "--device", type=str, default="mps", help="Device to use for the SAE."
    )

    return parser


def get_config_from_argparser(parser):
    args = parser.parse_args()
    # get/fix config
    config = get_wandb_config(args.wandb_id, args.wandb_entity, args.wandb_project)
    config = {**config, **vars(args)}
    config = argparse.Namespace(**config)
    return config



if __name__ == "__main__":
    parser = get_argparser()
    config = get_config_from_argparser(parser)
    record_save_path = create_neuron_records(config)
    # can be loaded like:
    # loaded_neuron_records = NeuronRecords.load(record_save_path)
