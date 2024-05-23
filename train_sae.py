import torch
import torch.nn.functional as F
import wandb

from datasets import load_dataset
from tqdm import tqdm

from src.paths import get_embeddings_cache_dir
from src.backbone import get_backbone
from src.act_dataset import ActivationDataset
from src.sae import SparseAutoEncoder
from src.orthogonal_sae import OrthogonalSAE
from src.utils import log_dict, wandb_log
from src.metrics import (
    mean_pairwise_cosine_similarity,
    mean_max_cosine_similarity,
    SlidingWindowDeadNeuronTracker,
)


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
    )

    del text_ds
    del backbone_model

    return ds


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

    dead_neuron_detector = SlidingWindowDeadNeuronTracker(len(dl), sae.M.shape[1])

    steps = 0

    for _ in range(args.epochs):
        sae.train()
        for x in (pbar := tqdm(dl)):
            metrics = {}
            optimizer.zero_grad()
            x = x.to(args.device)

            x_hat, c = sae(x)

            reconstruction_loss = F.mse_loss(x, x_hat)
            unscaled_sparsity_loss = torch.linalg.norm(c, ord=1, dim=-1).mean()
            sparsity_loss = args.l1 * unscaled_sparsity_loss
            loss = reconstruction_loss + sparsity_loss

            shear_loss = None
            # only aplicable for orthogonal
            if hasattr(sae, "shear_param") and sae.shear_param is not None:
                shear_loss = torch.linalg.norm(sae.shear_param, ord=1)
                metrics.update(
                    {
                        "train/unscaled_shear_loss": shear_loss.item(),
                        "train/scaled_shear_loss": shear_loss.item()
                        * args.shear_l1,
                    }
                )
                if args.shear_l1 > 0:
                    loss += args.shear_l1 * shear_loss

            loss.backward()
            optimizer.step()
            steps += args.batch_size

            neuron_inactivity_periods = dead_neuron_detector.on_batch(c)

            cos_sim = mean_pairwise_cosine_similarity(x, x_hat)

            # somewhat pricey to calulate
            mean_max_cos_D = mean_max_cosine_similarity(sae.D)

            metrics.update(
                {
                    "train/neuron_inactivity_periods@0.75": neuron_inactivity_periods.quantile(
                        0.75
                    ).item(),
                    "train/neuron_inactivity_periods@0.5": neuron_inactivity_periods.quantile(
                        0.5
                    ).item(),
                    "train/neuron_inactivity_periods@0.25": neuron_inactivity_periods.quantile(
                        0.25
                    ).item(),
                    "train/num_dead_neurons": (neuron_inactivity_periods == 1.0)
                    .sum()
                    .item(),
                    "train/loss": loss.item(),
                    "train/reconstruction_loss": reconstruction_loss.item(),
                    "train/reconstruction_cos_sim": cos_sim.item(),
                    "train/sparsity_loss": sparsity_loss.item(),
                    "train/unscaled_sparsity_loss": unscaled_sparsity_loss.item(),
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/steps": steps,
                    "train/mean_max_cos_D": mean_max_cos_D.item(),
                }
            )

            wandb_log(metrics)

            pbar.set_description(
                f"Loss: {loss.item():.4f}, Cosine Sim: {cos_sim.item():.4f}"
            )

        scheduler.step()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-70m-deduped")
    parser.add_argument("--cache_dir", type=str, default=get_embeddings_cache_dir())
    parser.add_argument("--text_dataset", type=str, default="NeelNanda/pile-10k")
    parser.add_argument("--layername", type=str, default="layers.4")

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
        "--tied", type=int, default=0, help="Tie the weights of the model"
    )
    # only applicable for orthogonal
    parser.add_argument(
        "--allow_shear",
        type=int,
        default=0,
        help="Allow shear in the decoder matrix",
    )

    # training
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=10)
    # losses
    parser.add_argument("--l1", type=float, default=1e-3)
    # only applicable for orthogonal
    parser.add_argument("--shear_l1", type=float, default=1e-3)

    # misc
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--wandb", type=str, default="", help="wandb project name")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # convert to bool (easier to use in wandb sweep if these are ints)
    args.tied = bool(args.tied)
    args.allow_shear = bool(args.allow_shear)

    set_seed(args.seed)
    main(args)
