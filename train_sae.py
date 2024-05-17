import torch
import wandb

from datasets import load_dataset
from tqdm import tqdm

from src.paths import get_embeddings_cache_dir
from src.backbone import get_backbone
from src.act_dataset import ActivationDataset
from src.sae import SparseAutoEncoder
from src.utils import log_dict, wandb_log
from src.metrics import DeadNeuronDetector, mean_cosine_similarity


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

    if args.info:
        for name, _ in backbone_model.named_modules():
            print(name)
        return None

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

    if ds is None:
        if args.info:
            return
        else:
            raise ValueError("Dataset is None")

    sae = SparseAutoEncoder(
        in_features=ds.data.shape[-1],
        hidden_dim=int(args.R * ds.data.shape[-1]),
        tied=args.tied,
        bias=True,
    )
    log_dict({f"model/M_shape_{i}": v for i, v in enumerate(sae.M.shape)}, config=True)

    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    sae.init_weights_D_(args.init_strategy_D)
    if not args.tied:
        sae.init_weights_M_(args.init_strategy_M)
    sae.to(args.device)
    sae.train()

    # in: [Interim research report] Taking features out of superposition with sparse autoencoders
    # lr = 0.001, batch_size = 256, optim = Adam
    optimizer = torch.optim.Adam(sae.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs + 1)

    dead_neuron_detector = DeadNeuronDetector()

    for epoch in range(args.epochs):
        sae.train()
        for x in (pbar := tqdm(dl)):
            optimizer.zero_grad()
            x = x.to(args.device)
            x_hat, c = sae(x)
            reconstruction_loss, sparsity_loss = sae.losses(x, c, x_hat, args.l1)
            loss = reconstruction_loss + sparsity_loss
            loss.backward()
            optimizer.step()

            dead_neuron_detector.on_batch(c)
            cos_sim = mean_cosine_similarity(x, x_hat)

            wandb_log(
                {
                    "train/loss": loss.item(),
                    "train/reconstruction_loss": reconstruction_loss.item(),
                    "train/reconstruction_cos_sim": cos_sim.item(),
                    "train/sparsity_loss": sparsity_loss.item(),
                    "train/unscaled_sparsity_loss": sparsity_loss.item() / args.l1,
                }
            )
            pbar.set_description(
                f"Loss: {loss.item():.4f}, Cosine Sim: {cos_sim.item():.4f}"
            )

        _, num_dead = dead_neuron_detector.on_epoch_end()
        log_dict(
            {
                "num_dead_neurons": num_dead,
                "after_epoch": epoch,
            }
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
        "--R", type=float, default=2.0, help="Multiplier for the hidden layer size"
    )
    parser.add_argument(
        "--init_strategy_M",
        type=str,
        default="xavier",
        help="Note: will be ignored if tied is True",
    )
    parser.add_argument("--init_strategy_D", type=str, default="xavier")

    # training
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--l1", type=float, default=1e-3)
    parser.add_argument(
        "--tied", type=int, default=0, help="Tie the weights of the model"
    )

    # misc
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--info", action="store_true")
    parser.add_argument("--wandb", type=str, default="", help="wandb project name")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    args.tied = bool(args.tied)

    set_seed(args.seed)
    main(args)
