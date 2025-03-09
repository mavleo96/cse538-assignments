#!/usr/bin/env python3

import os
import argparse
import csv

import numpy as np
from tqdm import tqdm
from transformers import GPT2TokenizerFast, PreTrainedTokenizerFast
from typing import List, Union

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ==========================
#       TrigramLM Class
# ==========================


class TrigramLM:
    """Trigram Language Model"""

    def __init__(self, tokenizer: PreTrainedTokenizerFast):
        """Initialize the TrigramLM"""
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size + 2  # for <s> and </s>
        assert self.vocab_size > 2
        # TODO: check if this is correct and if we need to add <|endoftext|> to the vocab
        self.unigram_counts = {}
        self.bigram_counts = {}
        self.trigram_counts = {}
        # TODO: check if defaultdict is safe here

    def train(self, data: List[List[str]]) -> None:
        """Train the TrigramLM"""
        # Tokenize the data
        tokenized_data = [self._tokenize(i) for i in data]

        # Loop through the tokenized data
        for row in tqdm(tokenized_data, desc="Training TrigramLM"):
            assert len(row) > 0
            # Add <s> and </s> tokens to the row
            row = ["<s>"] + row + ["</s>"]

            # Loop through the tokens in the row
            for j, _ in enumerate(row):
                # Count unigrams
                self.unigram_counts[row[j]] = self.unigram_counts.get(row[j], 0) + 1

                # Count bigrams
                if j > 0:
                    if row[j - 1] not in self.bigram_counts:
                        self.bigram_counts[row[j - 1]] = {}
                    self.bigram_counts[row[j - 1]][row[j]] = (
                        self.bigram_counts[row[j - 1]].get(row[j], 0) + 1
                    )

                # Count trigrams
                if j > 1:
                    if row[j - 2] not in self.trigram_counts:
                        self.trigram_counts[row[j - 2]] = {}
                    if row[j - 1] not in self.trigram_counts[row[j - 2]]:
                        self.trigram_counts[row[j - 2]][row[j - 1]] = {}
                    self.trigram_counts[row[j - 2]][row[j - 1]][row[j]] = (
                        self.trigram_counts[row[j - 2]][row[j - 1]].get(row[j], 0) + 1
                    )
                # TODO: check if defaultdict is safe here
        return None

    def nextProb(self, history_toks: List[str], next_toks: List[str]) -> float:
        """Compute the probability of the next token given the history"""
        assert isinstance(history_toks, list), "history_toks must be a list"
        assert isinstance(next_toks, list), "next_toks must be a list"
        assert len(next_toks) > 0, "next_toks must not be empty"

        # Case 1: No history
        if len(history_toks) == 0:
            # Compute unigram probabilities
            n_counts = [self.unigram_counts.get(tok, 0) for tok in next_toks]
            d_counts = sum(self.unigram_counts.values())

        # Case 2: One history token
        elif len(history_toks) == 1:
            # Compute bigram probabilities
            prev_tok = history_toks[0]
            n_counts = [
                self.bigram_counts.get(prev_tok, {}).get(tok, 0) for tok in next_toks
            ]
            d_counts = self.unigram_counts.get(prev_tok, 0)

        # Case 3: Two or more history tokens
        else:
            # Compute trigram probabilities
            prev_tok1, prev_tok2 = history_toks[-2:]
            n_counts = [
                self.trigram_counts.get(prev_tok1, {}).get(prev_tok2, {}).get(tok, 0)
                for tok in next_toks
            ]
            d_counts = self.bigram_counts.get(prev_tok1, {}).get(prev_tok2, 0)

        # Return the add-one smoothed probabilities
        return self._add_one_smoothed_prob(n_counts, d_counts)

    def _tokenize(self, text: str) -> List[str]:
        """Internal method to tokenize text"""
        return self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(text))

    def _add_one_smoothed_prob(
        self, n_counts: Union[int, List[int]], d_counts: int
    ) -> Union[float, List[float]]:
        """Internal method to compute add-one smoothed probabilities"""
        if isinstance(n_counts, int):
            return (n_counts + 1) / (d_counts + self.vocab_size)
        else:
            return [(n + 1) / (d_counts + self.vocab_size) for n in n_counts]


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

    # Initialize and train TrigramLM
    outfile.write("Checkpoint 1.2:\n")

    print("Initializing and training TrigramLM...")
    lmodel = TrigramLM(tokenizer)
    lmodel.train([i[2] for i in data])

    # Compute probabilities for given history and next tokens
    test_cases = [
        (["<s>", "Are", "Ġwe"], ["Ġout", "Ġin", "Ġto", "Ġpretending", "Ġonly"]),
        (["And", "ĠI"], ["Ġwas", "'m", "Ġstood", "Ġknow", "Ġscream", "Ġpromise"]),
    ]
    for history_toks, next_toks in test_cases:
        prob = lmodel.nextProb(history_toks, next_toks)
        prob = {next_toks[i]: f"{prob[i]:.2e}" for i in range(len(next_toks))}
        outfile.write(f"{prob}\n")
    outfile.write("\n")

    # Close output file
    print("Closing output file...")
    outfile.close()

    return None


if __name__ == "__main__":
    main()
