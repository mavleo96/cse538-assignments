#!/usr/bin/env python3

import os
import sys
import argparse
import re

from typing import List

# ==========================
#     Word Tokenizer
# ==========================


REGEX_PATTERN = r"""
https?://\S+\.\S+\w\/?|                              # URLs with http or https
\w+\.com\b|                                          # URLs with .com
[:;]-?[\)D\(P/]|                                     # Emoticons 1
[DP][:;]|                                            # Emoticons 2
(?:[A-Z]\.)+|                                        # Abbreviations
[A-z]+[`'][A-z]+|                                    # Contractions
\d+\.\d+|                                            # Numbers with decimal
\d+:\d+|                                             # Time
# [$£]?(?:\d{,3},)*\d+(?:\.\d+)?|                    # Money
\w+[\/]\w+|                                          # Words with slashes
(?:\.+|,+|!+|\?+|\(+|\)+|\?\!|[:;"'`~\{\}\[\]])|     # Punctuation
[@#]?[\w\-]+|                                        # Words with optional @ or #
\S                                                   # Any other non-whitespace character
"""
# TODO: A. should be captured as ["A", "."] and not ["A."]
# TODO: Need to check if money has to be captured
# TODO: 19 mismatch out of 150 with this regex


def wordTokenizer(sent: str) -> List[str]:
    """Split a string into list of tokens matched by regex"""
    pattern = re.compile(REGEX_PATTERN, re.VERBOSE)
    tokens = re.findall(pattern, sent)

    # Check if tokens add back to original sentence
    assert "".join(tokens) == "".join(
        sent.split()
    ), f"Tokens don't add up to original sentence\nTokens: {tokens}\nSentence: {sent}"
    return tokens


# ==========================
#     Byte Pair Encoding
# ==========================


def spacelessBPELearn(docs, max_vocabulary=1000):
    # input: docs, a list of strings to be used as the corpus for learning the BPE vocabulary
    # output: final_vocabulary, a set of all members of the learned vocabulary
    # non-ascii letter should be tagged as "?"
    final_vocabulary = None
    return final_vocabulary


def spacelessBPETokenize(text, vocab):
    # input: text, a single string to be word tokenized.
    #       vocab, a set of valid vocabulary words
    # output: words, a list of strings of all word tokens, in order, from the string
    words = None
    return words


# ==========================
#     Main Function
# ==========================


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
    print("Creating output file...")
    os.makedirs("results", exist_ok=True)
    outfile = open("results/a1_p1_murugan_116745378_OUTPUT.txt", "w")

    # Tokenize and print first 5 and last doc in input data
    outfile.write("Checkpoint 1.1:\n")
    print("Tokenizing first 5 and last doc in input data...")
    test_data = data[:5] + (data[-1:] if len(data) > 5 else [])
    for s in test_data:
        result = wordTokenizer(s)
        result_string = ",".join(result) + "\n"
        outfile.write(result_string)
    outfile.write("\n")

    outfile.write("Checkpoint 1.2:\n")

    # Close output file
    print("Closing output file...")
    outfile.close()

    return None


if __name__ == "__main__":
    main()
