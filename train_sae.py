import torch

from datasets import load_dataset
from tqdm import tqdm

from src.paths import get_embeddings_cache_dir
from src.backbone import get_backbone
from src.act_dataset import ActivationDataset
from src.sae import TiedSparseAutoEncoder


def get_ds(args):
    backbone_model = get_backbone(args.model)
    text_ds = load_dataset(args.text_dataset)['train']

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

    sae = TiedSparseAutoEncoder(ds.data.shape[-1], 2 * ds.data.shape[-1])
    sae.to(args.device)
    sae.train()
    
    dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)

    optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3)

    for epoch in range(10):
        losses = []
        for batch in (pbar:=tqdm(dl)):
            optimizer.zero_grad()
            batch = batch.to(args.device)
            _, loss, _ = sae(batch)
            loss.backward()
            optimizer.step()
            pbar.set_description(f"Loss: {loss.item():.4f}")
            losses.append(loss.item())
        print(f"Epoch {epoch} Loss: {sum(losses)/len(losses):.4f}")            

    



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="EleutherAI/pythia-70m-deduped")
    parser.add_argument("--cache_dir", type=str, default=get_embeddings_cache_dir())
    parser.add_argument("--text_dataset", type=str, default="NeelNanda/pile-10k")
    parser.add_argument("--layername", type=str, default="layers.5.mlp.act")
    parser.add_argument("--device", type=str, default="mps")

    args = parser.parse_args()

    main(args)