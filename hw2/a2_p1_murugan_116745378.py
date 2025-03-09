#!/usr/bin/env python3

import os
import argparse
import csv

from transformers import GPT2TokenizerFast
from typing import List


# ==========================
#     Main Function
# ==========================


def main() -> None:
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="script to run cse538 assignment 2 part 1"
    )
    parser.add_argument("filepath", type=str, help="path to the input file")
    args = parser.parse_args()

    # Read and process input data
    with open(args.filepath, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        data = [row for row in reader]

    # Create and open output file
    print("Creating output file...")
    os.makedirs("results", exist_ok=True)
    outfile = open("results/a2_p1_murugan_116745378_OUTPUT.txt", "w")

    # Tokenize using pre-trained GPT2 tokenizer
    outfile.write("Checkpoint 1.1:\n")

    print("Initializing GPT2 tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"
    tokenizer.pad_token = "<|endoftext|>"

    print("Tokenizing first and last row in input data using GPT2 tokenizer...")
    first_row = tokenizer.convert_ids_to_tokens(tokenizer.encode(data[0][2]))
    last_row = tokenizer.convert_ids_to_tokens(tokenizer.encode(data[-1][2]))

    outfile.write(f"first: {first_row}\n")
    outfile.write(f"last: {last_row}\n")
    outfile.write("\n")

    # Close output file
    print("Closing output file...")
    outfile.close()

    return None


if __name__ == "__main__":
    main()
