import os


def get_data_dir():
    p = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(p, exist_ok=True)
    return p


def get_embeddings_cache_dir():
    p = os.path.join(get_data_dir(), "embeddings_cache")
    os.makedirs(p, exist_ok=True)
    return p


def get_checkpoints_save_dir():
    p = os.path.join(get_data_dir(), "sae_checkpoints")
    os.makedirs(p, exist_ok=True)
    return p


def get_neuron_recods_save_dir():
    p = os.path.join(get_data_dir(), "neuron_records")
    os.makedirs(p, exist_ok=True)
    return p


def get_interpretability_save_dir():
    p = os.path.join(get_data_dir(), "interpretability")
    os.makedirs(p, exist_ok=True)
    return p
