import wandb


def wandb_log(data):
    try:
        wandb.log(data)
    except wandb.errors.Error:
        pass


def log_dict(data):
    wandb_log(data)
    for k, v in data.items():
        print(f"{k}: {v}")
