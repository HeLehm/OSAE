# helper script to run the sweep of the hyperparameters using wandb

"""
program: sweep.py
method: grid
metric:
  goal: minimize
  name: "train/mean_max_cos_D"
parameters:
  variant:
    values:
      - vanilla
      - vanilla_tied
      - orthogonal
      - orthogonal_shear
      - orthogonal_shear@0.001
      - orthogonal_shear@0.0001
  R:
    values:
      - 1
      - 2
  l1:
    values:
      - 0.001
      - 0.0001
  seed:
    values:
      - 42
      - 43
      - 44
  wandb:
    value: NNOrthogonal
"""

import argparse
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Your script description")
    parser.add_argument("--variant", type=str, help="Variant name", required=True)
    parser.add_argument("--tied_int", type=int, default=0, help="Tie the weights of the model")
    args, unknown_args = parser.parse_known_args()
    args.tied_int = bool(args.tied_int)

    cmd = [
        "python",
        "train_sae.py",
    ]

    if args.variant == "vanilla":
        cmd.extend(
            [
                "--architecture",
                "vanilla",
            ]
        )
    elif args.variant == "vanilla_tied":
        cmd.extend(["--architecture", "vanilla", "--tied"])

    elif args.variant == "orthogonal":
        cmd.extend(
            [
                "--architecture",
                "orthogonal",
            ]
        )
    elif args.variant.startswith("orthogonal_shear"):
        cmd.extend(["--architecture", "orthogonal", "--allow_shear"])
        if "@" in args.variant:
            shear_l1 = args.variant.split("@")[1]
            cmd.extend(["--shear_l1", shear_l1])
    else:
        raise ValueError(f"Unknown variant {args.variant}")

    cmd.extend(unknown_args)

    if args.tied_int:
        cmd.extend(["--tied"])

    print(f"Running command: {' '.join(cmd)}")
    # run the command
    subprocess.run(cmd)
