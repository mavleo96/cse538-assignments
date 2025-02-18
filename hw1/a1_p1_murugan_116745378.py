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
#     Prefix Tree (Vocab)
# ==========================


class TreeNode:
    """Prefix tree node"""

    def __init__(self):
        """Initialize a new tree node"""
        self.children = {}
        self.isword = False


class PrefixTree:
    """Prefix tree"""

    def __init__(self):
        """Initialize a new prefix tree"""
        self.root = TreeNode()

    def insert(self, word: str) -> None:
        """Insert a word into the prefix tree"""
        node = self.root

        # Loop through each letter
        for letter in word:
            # If letter is not in children, add it
            if letter not in node.children:
                node.children[letter] = TreeNode()
            node = node.children[letter]
        node.isword = True

    def longest_prefix(self, word: str) -> str:
        """Find the longest prefix of a word in the prefix tree"""
        node = self.root
        prefix, buffer = "", ""

        # Loop through each letter
        for letter in word:
            # If letter is in children, move to next node
            if letter in node.children:
                node = node.children[letter]
                # Add letter to buffer; if node is a word, add buffer to prefix and reset buffer
                buffer += letter
                if node.isword:
                    prefix += buffer
                    buffer = ""
            else:
                break
        return prefix


# ==========================
#     Byte Pair Encoding
# ==========================


def spacelessBPELearn(docs, max_vocabulary=1000):
    # input: docs, a list of strings to be used as the corpus for learning the BPE vocabulary
    # output: final_vocabulary, a set of all members of the learned vocabulary
    # non-ascii letter should be tagged as "?"
    final_vocabulary = [chr(i) for i in range(256)]
    return final_vocabulary


def spacelessBPETokenize(text, vocab):
    """Tokenize a string using a given vocabulary"""
    prefix_tree = PrefixTree()
    for v in vocab:
        prefix_tree.insert(v)

    tokens = []
    # Split text into words to skip spaces
    text = text.split()
    # Loop through each word and find longest prefix
    for word in text:
        while word:
            prefix = prefix_tree.longest_prefix(word)
            # If no prefix found, add "?" to tokens and move to next character
            if not prefix:
                tokens.append("?")
                word = word[1:]
            else:
                tokens.append(prefix)
                word = word[len(prefix) :]

    return tokens


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
    # Select first 5 and last doc in input data
    test_data = data[:5] + (data[-1:] if len(data) > 5 else [])

    # Create and open output file
    print("Creating output file...")
    os.makedirs("results", exist_ok=True)
    outfile = open("results/a1_p1_murugan_116745378_OUTPUT.txt", "w")

    # Tokenize using regex tokenizer
    outfile.write("Checkpoint 1.1:\n")
    print("Tokenizing first 5 and last doc in input data using regex...")
    outfile.write("Tokenized data:\n")
    for s in test_data:
        result = wordTokenizer(s)
        result_string = ",".join(result) + "\n"
        outfile.write(result_string)
    outfile.write("\n")

    # Tokenize using SLBPE tokenizer
    outfile.write("Checkpoint 1.2:\n")
    # Learn vocabulary from input data
    print("Learning vocabulary from input data using SLBPE...")
    outfile.write("Top 5 most frequent pairs:\n")
    final_vocabulary = spacelessBPELearn(data)
    outfile.write("\n")

    print("Writing final vocabulary to output file...")
    outfile.write("Final vocabulary:\n")
    outfile.write(",".join(final_vocabulary) + "\n\n")

    # Tokenize and print first 5 and last doc in input data
    print("Tokenizing first 5 and last doc in input data using SLBPE...")
    outfile.write("Tokenized data:\n")
    for s in test_data:
        result = spacelessBPETokenize(s, final_vocabulary)
        result_string = ",".join(result) + "\n"
        outfile.write(result_string)

    # Close output file
    print("Closing output file...")
    outfile.close()

    return None


if __name__ == "__main__":
    main()
