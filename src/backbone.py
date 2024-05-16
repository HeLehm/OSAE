from transformers import AutoModel, AutoTokenizer


def get_backbone(name):
    model = AutoModel.from_pretrained(name)
    if not hasattr(model, "tokenizer"):
        model.tokenizer = AutoTokenizer.from_pretrained(name)
    # add padding if needed
    if not hasattr(model.tokenizer, "pad_token") or model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
    return model
