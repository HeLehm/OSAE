from src.backbone import get_backbone
from src.paths import get_embeddings_cache_dir
from src.custom_hooked import HookedModel
from src.act_dataset import ActivationDataset
from datasets import load_dataset


def main(args):
    model = get_backbone(args.model)

    if args.info:
        print("Model Info:")
        for name, mod in model.named_modules():
            print(name)
        return
    
    if args.layername is None:
        raise ValueError("Please specify a layername")
    
    model = HookedModel(model)
    model.to(args.device)
    # TODO: dirty to get the split like this
    text_ds = load_dataset(args.text_dataset)['train']
    ActivationDataset(
        args.layername,
        model,
        text_ds,
        cache_root_dir=args.cache_dir,
    )


    
    

    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="EleutherAI/pythia-70m-deduped")
    parser.add_argument("--cache_dir", type=str, default=get_embeddings_cache_dir())
    parser.add_argument("--text_dataset", type=str, default="NeelNanda/pile-10k")
    parser.add_argument("--layername", type=str, default="layers.5.mlp.act")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--info", action="store_true")

    args = parser.parse_args()

    main(args)

