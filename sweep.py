# helper script to run the sweep of the hyperparameters using wandb

import argparse
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Your script description')
    parser.add_argument('--variant', type=str, help='Variant name', required=True)
    args, unknown_args = parser.parse_known_args()

    cmd = [
        "python", "train_sae.py",
    ]
    
    if args.variant == "vanilla":
        cmd.extend(
            [
                "--architecture", "vanilla",
            ]
        )
    elif args.variant == "vanilla_tied":
        cmd.extend(
            [
                "--architecture", "vanilla",
                "--tied"
            ]
        )

    elif args.variant == "orthogonal":
        cmd.extend(
            [
                "--architecture", "orthogonal",
            ]
        )
    elif args.variant.startswith("orthogonal_shear"):
        cmd.extend(
            [
                "--architecture", "orthogonal",
                "--allow_shear"
            ]
        )
        if "@" in args.variant:
            shear_l1 = args.variant.split("@")[1]
            cmd.extend(
                [
                    "--shear_l1", shear_l1
                ]
            )
    else:
        raise ValueError(f"Unknown variant {args.variant}")
    
    cmd.extend(unknown_args)

    print(f"Running command: {' '.join(cmd)}")
    # run the command
    subprocess.run(cmd)
