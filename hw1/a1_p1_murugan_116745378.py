#!/usr/bin/env python3

import os
import argparse
import re

from typing import List
from collections import Counter
from tqdm import tqdm


# ==========================
#     Word Tokenizer
# ==========================


REGEX_PATTERN = r"""
https?://\S+\.\S+\w/?|                               # URLs with http or https
\w+\.com\b|                                          # URLs with .com
[:;]-?[\)D\(P/]| [DP][:;]|                           # Emoticons
(?:[A-Z]\.)+|                                        # Abbreviations
[A-Za-z]+[`'][A-Za-z]+|                              # Contractions
\d+\.\d+|                                            # Numbers with decimal
\d+:\d+|                                             # Time
# [$£]?(?:\d{1,3},)*\d+(?:\.\d+)?[kKmM]?|              # Money
\w+/\w+|                                             # Words with slashes
(?:\.+|,+|!+|\?+|\(+|\)+|\?\!|[:;"'`~\{\}\[\]])|     # Punctuation
[@#]?[\w\-]+|                                        # Words with optional @ or #
# [@#]?(?:\w[\w-]+)*\w+|                               # Words with optional @ or #
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


def spacelessBPELearn(docs, max_vocabulary=1000):
    """Learn a vocabulary from a list of documents"""
    global outfile
    # Initialize vocabulary with all ascii characters
    N_ASCII = 256
    vocab = [chr(i) for i in range(N_ASCII)]

    # Initialize word count with all words in docs
    word_count = Counter(i for d in docs for i in d.split() if len(i) > 1)
    tokenized_words = {word: spacelessBPETokenize(word, vocab) for word in word_count}

    # Loop until vocabulary size is reached
    for iter in tqdm(range(max_vocabulary - N_ASCII)):
        purge = set()

        # Loop through each word and find all pairs
        pairs = Counter()
        for word, freq in word_count.items():
            tokens = tokenized_words[word]
            # If word is already a single token, mark for removal
            if len(tokens) <= 1:
                purge.add(word)
                continue
            for i in range(len(tokens) - 1):
                # Skip pairs with "?"
                if tokens[i] == "?" or tokens[i + 1] == "?":
                    continue
                pairs[tokens[i], tokens[i + 1]] += freq

        # Remove words with length <= 1 from word count
        for word in purge:
            del word_count[word]
            del tokenized_words[word]

        # Find most common pair and add it to vocabulary
        mc_pair = pairs.most_common(1)[0][0]
        vocab.append("".join(mc_pair))

        # Update word count and tokenized words
        contains_pair = lambda tokens, pair: any(
            tokens[i] == pair[0] and tokens[i + 1] == pair[1]
            for i in range(len(tokens) - 1)
        )
        for word in word_count:
            if contains_pair(tokenized_words[word], mc_pair):
                tokenized_words[word] = spacelessBPETokenize(word, vocab)

        # Write top 5 pairs to output file
        if iter in [0, 1, 10, 100, 500]:
            outfile.write(f"iter {iter}: {pairs.most_common(5)}\n")

    return vocab


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
    global outfile
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
