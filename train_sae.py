import torch
import wandb

from datasets import load_dataset
from tqdm import tqdm

from src.paths import get_embeddings_cache_dir
from src.backbone import get_backbone
from src.act_dataset import ActivationDataset
from src.sae import TiedSparseAutoEncoder
from src.utils import log_dict, wandb_log


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
        for name, module in backbone_model.named_modules():
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


@torch.no_grad()
def evaluate_sae(sae, dl, device, l1_coeff=1e-3):
    sae.eval()
    metrics = []
    for batch in tqdm(dl, desc="Evaluating"):
        batch = batch.to(device)
        x_hat, reconstruction_loss, sparsity_loss = sae(batch, l1_coeff)
        cos_sim = torch.nn.functional.cosine_similarity(x_hat, batch).mean()
        metrics.append(
            {
                "reconstruction_loss": reconstruction_loss.item(),
                "cos_sim": cos_sim.item(),
                "scaled_sparsity_loss": sparsity_loss.item(),
            }
        )
    mean_metrics = {
        "mean_" + k: sum(m[k] for m in metrics) / len(metrics) for k in metrics[0]
    }
    return mean_metrics


def main(args):
    if args.wandb:
        wandb.init(project="SAE", config=args)

    ds = get_ds(args)

    if ds is None:
        if args.info:
            return
        else:
            raise ValueError("Dataset is None")

    sae = TiedSparseAutoEncoder(ds.data.shape[-1], args.R * ds.data.shape[-1])
    print("Model M weight shape:", sae.M.shape)
    log_dict({f"model/M_shape_{i}": v for i, v in enumerate(sae.M.shape)})

    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    sae.init_weights(args.init_strategy, dl)
    sae.to(args.device)
    sae.train()

    # don't use Adam, it works to fast, so its quite hard to compare the results
    optimizer = torch.optim.AdamW(sae.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs + 1)

    for epoch in range(args.epochs):
        eval_metrics = evaluate_sae(sae, dl, args.device, l1_coeff=args.l1_coeff)
        log_dict(
            {
                **{f"eval/{k}": v for k, v in eval_metrics.items()},
                "before_epoch": epoch,
            }
        )
        for batch in (pbar := tqdm(dl)):
            optimizer.zero_grad()
            batch = batch.to(args.device)
            x_hat, reconstruction_loss, sparsity_loss = sae(
                batch, l1_coeff=args.l1_coeff
            )
            loss = reconstruction_loss + sparsity_loss
            loss.backward()
            optimizer.step()
            cos_sim = torch.nn.functional.cosine_similarity(x_hat, batch).mean()
            wandb_log(
                {
                    "train/reconstruction_loss": reconstruction_loss.item(),
                    "train/cos_sim": cos_sim.item(),
                    "train/sparsity_loss": sparsity_loss.item(),
                }
            )
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
        "--R", type=int, default=2, help="Multiplier for the hidden layer size"
    )
    parser.add_argument("--init_strategy", type=str, default="xavier")

    # training
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--l1_coeff", type=float, default=1e-3)

    # misc
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--info", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)
    main(args)
