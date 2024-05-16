import wandb


def wandb_log(data):
    try:
        wandb.log(data)
    except wandb.errors.Error:
        pass
