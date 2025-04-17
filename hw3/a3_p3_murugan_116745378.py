#! /usr/bin/env python3

import argparse
import os

import torch
import torch.nn as nn

torch.manual_seed(0)


# ==========================
#     Main Function
# ==========================


def main() -> None:
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="script to run cse538 assignment 3 part 3"
    )
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--save_model", action="store_true", default=False)
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--file_prefix", type=str, default="a3_p3_murugan_116745378")
    parser.add_argument("--use_subset", action="store_true", default=False)
    args = parser.parse_args()
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Create and open output file
    print("Creating output file...")
    os.makedirs(args.save_dir, exist_ok=True)
    outfile = open(f"{args.save_dir}/{args.file_prefix}_OUTPUT.txt", "w")

    # Close output file
    print("Closing output file...")
    outfile.close()

    return None


if __name__ == "__main__":
    main()
