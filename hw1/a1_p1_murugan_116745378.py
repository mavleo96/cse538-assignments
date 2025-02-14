#!/usr/bin/env python3

import sys
import argparse
import re

from typing import List


def wordTokenizer(sent: str) -> List[str]:
    """Split a string into list of tokens matched by regex"""
    # TODO: Need to check if the regex is accurate enough
    # TODO: A. should be captured as ["A", "."] and not ["A."]
    pattern = re.compile(
        r"(?:[A-Z]\.)+|[A-z]+'[A-z]+|\d+\.\d+|[.,:;'`]|[@#]?[A-Za-z0-9]+|\S+"
    )
    tokens = re.findall(pattern, sent)

    # Check if tokens add back to original sentence
    assert "".join(tokens) == "".join(
        sent.split()
    ), f"Tokens don't add up to original sentence\nTokens: {tokens}\nSentence: {sent}"
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
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="script to run cse538 assignment 1 part 1"
    )
    parser.add_argument("filepath", type=str, help="path to the input file")
    args = parser.parse_args()

    # Read and process input data
    with open(args.filepath, "r") as f:
        data = f.read().splitlines()

    # Create and open output file
    outfile = open("a1_p1_murugan_116745378_OUTPUT.txt", "w")

    # Tokenize and print first 5 and last doc in input data
    outfile.write("Checkpoint 1.1:\n")
    test_data = data[:5] + (data[-1:] if len(data) > 5 else [])
    for s in test_data:
        result = wordTokenizer(s)
        result_string = ",".join(result) + "\n"
        outfile.write(result_string)

    outfile.write("Checkpoint 1.2:\n")

    # Close output file
    outfile.close()

    return None


if __name__ == "__main__":
    main()
else:
    sys.exit(0)
