from transformer_lens import HookedTransformer
from transformers import AutoTokenizer


def get_model_and_tokenizer(name="EleutherAI/pythia-70m-deduped"):
    """
    Get the model and tokenizer from the model name
    Parameters
    ----------
    name : str
        The model name to load
        Default: "EleutherAI/pythia-70m-deduped"
        Another option: "EleutherAI/pythia-1.4B-deduped" (bigger model, slower inference)
    Returns
    -------
    model : HookedTransformer
        The model
    tokenizer : transformers.AutoTokenizer
        The tokenizer
    """
    model = HookedTransformer.from_pretrained(name)

    if hasattr(model, "tokenizer"):
        tokenizer = model.tokenizer
    else:
        print("Using default tokenizer from gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    return model, tokenizer