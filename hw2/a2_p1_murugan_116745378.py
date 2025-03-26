#!/usr/bin/env python3

import os
import argparse
import csv

import numpy as np
from tqdm import tqdm
from transformers import GPT2TokenizerFast, PreTrainedTokenizerFast
from typing import List, Union
from collections import defaultdict

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ==========================
#       TrigramLM Class
# ==========================


class TrigramLM:
    """Trigram Language Model"""

    def __init__(self, tokenizer: PreTrainedTokenizerFast):
        """Initialize the TrigramLM"""
        assert issubclass(
            type(tokenizer), PreTrainedTokenizerFast
        ), "tokenizer must be a PreTrainedTokenizerFast"
        self.tokenizer = tokenizer
        self.unk_token = tokenizer.unk_token
        self.vocab = tokenizer.vocab
        self.vocab_size = len(tokenizer.vocab)
        assert self.vocab_size > 0, "vocab_size must be greater than 0"

        # Intialize the n-gram counts
        self.unigram_count = defaultdict(int)
        self.bigram_count = defaultdict(lambda: defaultdict(int))
        self.trigram_count = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    def train(self, data: List[List[str]]) -> None:
        """Train the TrigramLM"""
        # Tokenize the data
        # Out of vocabulary tokens are handled by the tokenizer
        # and mapped to the unk_token
        tokenized_data = [self._tokenize(i) for i in data]

        # Loop through the tokenized data
        for row in tqdm(tokenized_data, desc="Training TrigramLM"):
            for j, _ in enumerate(row):
                # Count unigrams
                self.unigram_count[row[j]] += 1

                # Count bigrams
                if j > 0:
                    self.bigram_count[row[j - 1]][row[j]] += 1

                # Count trigrams
                if j > 1:
                    self.trigram_count[row[j - 2]][row[j - 1]][row[j]] += 1

        return None

    def nextProb(self, history_toks: List[str], next_toks: List[str]) -> float:
        """Compute the probability of the next token given the history"""
        assert isinstance(history_toks, list), "history_toks must be a list"
        assert isinstance(next_toks, list), "next_toks must be a list"
        assert len(next_toks) > 0, "next_toks must not be empty"

        # Handling OOV tokens
        # Case 1: If token is not in vocab, it is mapped to unk_token by tokenizer
        #         but since the tokens are arguments directly, we meed to assert that
        # Case 2: If token is in vocab, but not in counts, then it is mapped to unk_token here
        # Both cases are handled by membership check in unigram_count
        history_toks = [
            self.unk_token if tok not in self.unigram_count else tok
            for tok in history_toks
        ]
        next_toks = [
            self.unk_token if tok not in self.unigram_count else tok
            for tok in next_toks
        ]

        # Case 1: Unigram probabilities for one or no history tokens
        if len(history_toks) < 2:
            n_count = [self.unigram_count[tok] for tok in next_toks]
            d_count = sum(self.unigram_count.values())

        # Case 2: Trigram probabilities for two or more history tokens
        else:
            prev_tok1, prev_tok2 = history_toks[-2:]
            n_count = [
                self.trigram_count[prev_tok1][prev_tok2][tok] for tok in next_toks
            ]
            d_count = self.bigram_count[prev_tok1][prev_tok2]

        # Return the add-one smoothed probabilities
        return self._add_one_smoothed_prob(n_count, d_count)

    def get_sequence_probability(self, sequence: List[str]) -> List[float]:
        """Get the probability of the sequence"""
        probs = []
        for i in range(len(sequence)):
            history_toks = sequence[:i]
            next_toks = sequence[i : i + 1]
            probs.extend(self.nextProb(history_toks, next_toks))

        assert len(probs) == len(sequence), "probs and sequence must have same length"
        return probs

    def _tokenize(self, text: str) -> List[str]:
        """Internal method to tokenize text"""
        assert len(text) > 0, "text must not be empty"
        # Add <s> and </s> tokens to the row
        return ["<s>"] + self.tokenizer.tokenize(text) + ["</s>"]

    def _add_one_smoothed_prob(self, n_count: List[int], d_count: int) -> List[float]:
        """Internal method to compute add-one smoothed probabilities"""
        return [(n + 1) / (d_count + self.vocab_size) for n in n_count]


# ==========================
#        Perplexity
# ==========================


def get_perplexity(probs: List[float]) -> float:
    """Get the perplexity of the probabilities"""
    assert isinstance(probs, list), "probs must be a list"
    assert len(probs) > 0, "probs must not be empty"

    log_probs = np.log(probs)
    log_perplexity = (-1 / len(log_probs)) * np.sum(log_probs)
    perplexity = np.exp(log_perplexity)

    return perplexity


# ==========================
#      Helper Functions
# ==========================


def init_tokenizer() -> PreTrainedTokenizerFast:
    """Initialize the tokenizer"""
    tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
    tokenizer.add_special_tokens(
        {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "<|endoftext|>",
        }
    )

    return tokenizer


# ==========================
#        Observations
# ==========================

OBSERVATIONS = """Observations:
We see that the perplexity for the first and third test cases are lower than the rest.
This is because atleast one sequence of words in these test cases have appeared in the training data
while all the sequences in rest of the test cases seem to be relatively unseen.
"""
# TODO: Update observations

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
    with open(args.filepath, "r", newline="") as f:
        reader = csv.reader(f)
        data = list(reader)[1:-5]

    # Create and open output file
    print("Creating output file...")
    os.makedirs("results", exist_ok=True)
    outfile = open("results/a2_p1_murugan_116745378_OUTPUT.txt", "w")

    # Tokenize using pre-trained GPT2 tokenizer
    outfile.write("Checkpoint 1.1:\n")

    print("Initializing GPT2 tokenizer...")
    tokenizer = init_tokenizer()

    print("Tokenizing first and last row of training data using GPT2 tokenizer...")
    first_row = ["<s>"] + tokenizer.tokenize(data[0][2]) + ["</s>"]
    last_row = ["<s>"] + tokenizer.tokenize(data[-1][2]) + ["</s>"]

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

    # Compute perplexity
    outfile.write("Checkpoint 1.3:\n")

    print("Computing perplexity...")
    test_cases = [
        ["And", "Ġyou", "Ġgotta", "Ġlive", "Ġwith", "Ġthe", "Ġbad", "Ġblood", "Ġnow"],
        ["Sit", "Ġquiet", "Ġby", "Ġmy", "Ġside", "Ġin", "Ġthe", "Ġshade"],
        [
            "And",
            "ĠI",
            "'m",
            "Ġnot",
            "Ġeven",
            "Ġsorry",
            ",",
            "Ġnights",
            "Ġare",
            "Ġso",
            "Ġstar",
            "ry",
        ],
        [
            "You",
            "Ġmake",
            "Ġme",
            "Ġcraz",
            "ier",
            ",",
            "Ġcraz",
            "ier",
            ",",
            "Ġcraz",
            "ier",
            ",",
            "Ġoh",
        ],
        ["When", "Ġtime", "Ġstood", "Ġstill", "Ġand", "ĠI", "Ġhad", "Ġyou"],
    ]
    for seq in test_cases:
        probs = lmodel.get_sequence_probability(seq)
        perplexity = get_perplexity(probs)
        outfile.write(f"'{''.join(seq).replace('Ġ', ' ')}': {perplexity:.2f}\n")
    outfile.write("\n")

    # Observations
    outfile.write(OBSERVATIONS)

    # Close output file
    print("Closing output file...")
    outfile.close()

    return None


if __name__ == "__main__":
    main()
