import torch

from datasets import load_dataset
from tqdm import tqdm

from src.paths import get_embeddings_cache_dir
from src.backbone import get_backbone
from src.act_dataset import ActivationDataset
from src.sae import TiedSparseAutoEncoder


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


def main(args):
    ds = get_ds(args)

    if ds is None:
        if args.info:
            return
        else:
            raise ValueError("Dataset is None")

    sae = TiedSparseAutoEncoder(ds.data.shape[-1], args.R * ds.data.shape[-1])
    print("Model M weight shape:", sae.M.shape)
    sae.to(args.device)
    sae.train()

    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(sae.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs + 1)

    for epoch in range(args.epochs):
        metrics = []
        for batch in (pbar := tqdm(dl)):
            optimizer.zero_grad()
            batch = batch.to(args.device)
            x_hat, loss, sparsity_loss = sae(batch, l1_coeff=args.l1_coeff)
            loss.backward()
            optimizer.step()
            cos_sim = torch.nn.functional.cosine_similarity(x_hat, batch).mean()
            metrics.append(
                {
                    "loss": loss.item(),
                    "cos_sim": cos_sim.item(),
                    "sparsity_loss": sparsity_loss.item(),
                }
            )
            pbar.set_description(
                f"Loss: {loss.item():.4f}, Cosine Sim: {cos_sim.item():.4f}"
            )
        scheduler.step()

        mean_metrics = {
            k: sum(m[k] for m in metrics) / len(metrics) for k in metrics[0]
        }
        print(f"Epoch {epoch}")
        print(mean_metrics)


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

    # training
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--l1_coeff", type=float, default=1e-3)

    # misc
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--info", action="store_true")

    args = parser.parse_args()

    main(args)
