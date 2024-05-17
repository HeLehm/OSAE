import wandb


def wandb_log(data):
    try:
        wandb.log(data)
    except wandb.errors.Error:
        pass


def wandb_config_log(data):
    try:
        wandb.config.update(data)
    except wandb.errors.Error:
        pass


def log_dict(data, config=False):
    if not config:
        wandb_log(data)
    else:
        wandb_config_log(data)
    for k, v in data.items():
        print(f"{k}: {v}")
