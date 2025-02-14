#!/usr/bin/env python3

import sys
import argparse


def wordTokenizer(sent):
    # input: a single sentence as a string.
    # output: a list of each “word” in the text
    # must use regular expressions

    # <FILL IN>
    tokens = None
    return tokens


def spacelessBPELearn(docs, max_vocabulary=1000):
    # input: docs, a list of strings to be used as the corpus for learning the BPE vocabulary
    # output: final_vocabulary, a set of all members of the learned vocabulary
    final_vocabulary = None
    return final_vocabulary


def spacelessBPETokenize(text, vocab):
    # input: text, a single string to be word tokenized.
    #       vocab, a set of valid vocabulary words
    # output: words, a list of strings of all word tokens, in order, from the string
    words = None
    return words


def main() -> None:
    # parse arguments
    parser = argparse.ArgumentParser(
        description="script to run cse538 assignment 1 part 1"
    )
    parser.add_argument("filepath", type=str, help="path to the input file")
    args = parser.parse_args()

    # read and process input data
    with open(args.filepath, "r") as f:
        data = f.read().splitlines()

    print("Checkpoint 1.1:")

    print("Checkpoint 1.2:")

    return None


if __name__ == "__main__":
    main()
else:
    sys.exit(0)
