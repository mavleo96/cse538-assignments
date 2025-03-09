#!/usr/bin/env python3

import os
import argparse
import csv
import torch
import re

from torch.utils.data import TensorDataset, DataLoader
from transformers import GPT2TokenizerFast, PreTrainedTokenizerFast
from typing import List

# ==========================
#     Helper Functions
# ==========================


def chunk_tokens(
    tokens: List[int],
    start_token_id: int,
    end_token_id: int,
    pad_token_id: int,
    chunk_len: int = 128,
) -> torch.Tensor:
    """Chunk tokens into chunks of length chunk_len"""
    u_chunk_len = chunk_len - 2  # Remove start and end tokens from chunk length
    assert u_chunk_len > 0, "Chunk length must be greater than 2"

    # Pad tokens if necessary
    pad_count = (
        (u_chunk_len - len(tokens) % u_chunk_len)
        if len(tokens) % u_chunk_len != 0
        else 0
    )
    tokens += [pad_token_id] * pad_count

    # Chunk tokens by reshaping
    chunked_tokens = torch.tensor(tokens).reshape(-1, u_chunk_len)
    n_chunks = chunked_tokens.shape[0]

    # Concatenate BOS and EOS tokens
    bos_tensor = torch.full((n_chunks, 1), start_token_id)
    eos_tensor = torch.full((n_chunks, 1), end_token_id)
    chunks = torch.cat((bos_tensor, chunked_tokens, eos_tensor), dim=1)

    return chunks


def process_data(
    data: List[List[str]], tokenizer: PreTrainedTokenizerFast
) -> torch.Tensor:
    """Process, tokenize, and chunk the data"""
    PATTERN = r"\n\[[\x20-\x7f]+\]"

    # Remove section markers and tokenize the data
    data = [re.sub(PATTERN, "", row[2]) for row in data]
    tokenized_data = [tokenizer.encode(row) for row in data]

    # Chunk the tokens
    chunked_data = [
        chunk_tokens(
            tokens,
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
            tokenizer.pad_token_id,
            64,
        )
        for tokens in tokenized_data
    ]

    # Concatenate the chunks
    processed_data = torch.cat(chunked_data, dim=0)

    return processed_data


# ==========================
#     Main Function
# ==========================


def main() -> None:
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="script to run cse538 assignment 2 part 2"
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
    outfile = open("results/a2_p2_murugan_116745378_OUTPUT.txt", "w")
    # TODO: check if pdf output can be directly created instead

    # Close output file
    print("Closing output file...")
    outfile.close()

    return None


if __name__ == "__main__":
    main()
