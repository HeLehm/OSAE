import os
import torch
import torch.nn.functional as F
import wandb

from datasets import load_dataset
from tqdm import tqdm

from src.paths import get_embeddings_cache_dir, get_checkpoints_save_dir
from src.backbone import get_backbone
from src.act_dataset import ActivationDataset
from src.sae import SparseAutoEncoder
from src.orthogonal_sae import OrthogonalSAE
from src.utils import log_dict, wandb_log
from src.metrics import (
    mean_pairwise_cosine_similarity,
    mean_max_cosine_similarity,
    DeadNeuronDetector,
)
from typing import Optional, Callable, Dict


def set_seed(seed):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_ds(args):
    backbone_model = get_backbone(args.model)

    text_ds = load_dataset(args.text_dataset)["train"]

    ds = ActivationDataset(
        args.layername,
        backbone_model,
        text_ds,
        cache_root_dir=args.cache_dir,
        flatten_sequence=True,
        max_length=args.max_seq_length,
    )

    del text_ds
    del backbone_model

    return ds


def iterate_one_epoch(
    dl,
    sae,
    args,
    pre_forward: Optional[Callable[[None], None]] = lambda: None,
    additional_metrics: Optional[
        Callable[[torch.Tensor, torch.Tensor, torch.Tensor], Dict[str, float]]
    ] = lambda x, y, z: {},
    on_loss: Optional[Callable[[torch.Tensor], None]] = lambda x: None,
    on_batch_metrics: Optional[Callable[[Dict[str, float]], None]] = lambda x: None,
):
    avg_metrics = {}

    for x, _ in (pbar := tqdm(dl)):
        pre_forward()

        metrics = {}
        x = x.to(args.device)

        # forward pass
        x_hat, c = sae(x)

        # calculate losses
        reconstruction_loss = F.mse_loss(x, x_hat)
        unscaled_sparsity_loss = torch.linalg.norm(c, ord=1, dim=-1).mean()
        sparsity_loss = args.l1 * unscaled_sparsity_loss
        loss = reconstruction_loss + sparsity_loss

        # only aplicable for orthogonal
        if hasattr(sae, "shear_param") and sae.shear_param is not None:
            shear_loss = torch.linalg.norm(sae.shear_param, ord=1)
            metrics.update(
                {
                    "unscaled_shear_loss": shear_loss.item(),
                    "scaled_shear_loss": shear_loss.item() * args.shear_l1,
                }
            )
            if args.shear_l1 > 0:
                loss += args.shear_l1 * shear_loss

        metrics.update(
            {
                "reconstruction_loss": reconstruction_loss.item(),
                "unscaled_sparsity_loss": unscaled_sparsity_loss.item(),
                "sparsity_loss": sparsity_loss.item(),
                "loss": loss.item(),
            }
        )

        # for loss.backward and optimizer.step and such
        on_loss(loss)

        # update metrics
        metrics.update(additional_metrics(x, x_hat, c))

        # update average metrics
        for k, v in metrics.items():
            avg_metrics[k] = avg_metrics.get(k, 0) + v

        on_batch_metrics(metrics)

        pbar.set_description(f"Loss: {loss.item():.4f}")

    # average metrics
    for k, v in avg_metrics.items():
        avg_metrics[k] = v / len(dl)
    return avg_metrics


@torch.no_grad()
def evaluate_one_epoch(dl, sae, args):
    sae.eval()

    dnd = DeadNeuronDetector()

    def additional_eval_metrics(x, x_hat, c):
        cos_sim = mean_pairwise_cosine_similarity(x, x_hat)
        dnd.on_batch(c)
        return {
            "reconstruction_cos_sim": cos_sim.item(),
        }

    eval_metrics = iterate_one_epoch(
        dl, sae, args, additional_metrics=additional_eval_metrics
    )

    # add cosine sim of sae to eval_metrics
    eval_metrics["mean_max_cos_D"] = mean_max_cosine_similarity(sae.D).item()
    # add deat neuron_count
    _, dead_count = dnd.on_epoch_end()
    eval_metrics["dead_neurons_count"] = dead_count

    eval_metrics = {f"eval/ave_epoch/{k}": v for k, v in eval_metrics.items()}
    wandb_log(eval_metrics)

    return eval_metrics


def train_one_epoch(dl, sae, optimizer, args):
    sae.train()

    def on_loss(loss):
        loss.backward()
        optimizer.step()

    def log_train_metrics(metrics):
        wandb_log({f"train/batch/{k}": v for k, v in metrics.items()})

    def additional_train_metrics(x, x_hat, c):
        cos_sim = mean_pairwise_cosine_similarity(x, x_hat)
        return {
            "reconstruction_cos_sim": cos_sim.item(),
        }

    def pre_forward():
        optimizer.zero_grad()

    train_metrics = iterate_one_epoch(
        dl,
        sae,
        args,
        pre_forward=pre_forward,
        on_loss=on_loss,
        on_batch_metrics=log_train_metrics,
        additional_metrics=additional_train_metrics,
    )

    train_metrics = {f"train/ave_epoch/{k}": v for k, v in train_metrics.items()}
    wandb_log(train_metrics)

    return train_metrics


def main(args):
    if args.wandb != "":
        wandb.init(project=args.wandb, config=args, entity="bschergen")

    ds = get_ds(args)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    # get the constructor for the SAE
    sae_cls_map = {
        "vanilla": SparseAutoEncoder,
        "orthogonal": OrthogonalSAE,
    }
    sae_cls = sae_cls_map[args.architecture]

    # create the SAE
    sae = sae_cls(
        in_features=ds.data.shape[-1],
        hidden_dim=int(args.R * ds.data.shape[-1]),
        tied=args.tied,
        allow_shear=args.allow_shear,
        bias=True,
    )
    sae.to(args.device)
    sae.train()

    log_dict({f"model/M_shape_{i}": v for i, v in enumerate(sae.M.shape)}, config=True)

    # in: [Interim research report] Taking features out of superposition with sparse autoencoders
    # lr = 0.001, batch_size = 256, optim = Adam
    optimizer = torch.optim.Adam(sae.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs + 1)

    for _ in range(args.epochs):
        # log lr
        wandb_log({"train/lr": scheduler.get_last_lr()[0]})

        # train
        train_one_epoch(dl, sae, optimizer, args)

        scheduler.step()

    # evaluate
    evaluate_one_epoch(dl, sae, args)

    # save the model
    save_sae(sae, args)


def save_sae(sae, args):
    if args.save_path is not None:
        if args.save_path != "auto":
            sae.save(args.save_path)
        else:
            save_dir = get_checkpoints_save_dir()
            assert (
                args.wandb != ""
            ), "wandb project name must be provided, if save_path is auto"
            save_path = os.path.join(save_dir, wandb.run.id + ".pth")
            sae.save(save_path)


def get_args():
    import argparse

    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-70m-deduped")
    parser.add_argument("--cache_dir", type=str, default=get_embeddings_cache_dir())
    parser.add_argument("--text_dataset", type=str, default="NeelNanda/pile-10k")
    parser.add_argument("--layername", type=str, default="layers.4")
    parser.add_argument("--max_seq_length", type=int, default=128)

    # model
    parser.add_argument(
        "--architecture",
        type=str,
        default="vanilla",
        help="vanilla or orthogonal",
        choices=["vanilla", "orthogonal"],
    )
    parser.add_argument(
        "--R", type=float, default=2.0, help="Multiplier for the hidden layer size"
    )
    # only applicable for vanilla
    parser.add_argument(
        "--tied", action="store_true", help="Tie the weights of the model"
    )
    # only applicable for orthogonal
    parser.add_argument(
        "--allow_shear",
        action="store_true",
        help="Allow shear in the decoder matrix",
    )

    # training
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=30)
    # losses
    parser.add_argument("--l1", type=float, default=1e-3)
    # only applicable for orthogonal
    parser.add_argument("--shear_l1", type=float, default=0.0)

    # misc
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--wandb", type=str, default="", help="wandb project name")
    parser.add_argument("--seed", type=int, default=42)

    # save SAE
    parser.add_argument("--save_path", type=str, default=None)

    args, unknown_args = parser.parse_known_args()

    print(f"Unknown args: {unknown_args}")

    set_seed(args.seed)

    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
