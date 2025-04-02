#!/usr/bin/env python3

import argparse
import os
import re
from collections import Counter
from itertools import pairwise
from typing import List

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
\w+/\w+|                                             # Words with slashes
(?:\.+|,+|!+|\?+|\(+|\)+|\?\!|[:;"'`~\{\}\[\]])|     # Punctuation
[@#][\w\-]+|                                         # Words with optional @ or #
[A-Za-z0-9]+|                                        # Words
\S                                                   # Any other non-whitespace character
"""
# TODO: 34 mismatch out of 150 with this regex


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

    # Replace non-ascii characters with "?" in word string
    text = "".join([i if 32 <= ord(i) < 127 else "?" for i in text])
    # Split text into words to skip spaces
    text = text.split()

    # Loop through each word and find longest prefix
    tokens = []
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
    # Initialize vocabulary with all ascii characters
    N_ASCII = 127 - 33
    vocab = [chr(i) for i in range(33, 127)]

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
            for pair in pairwise(tokens):
                pairs[pair] += freq

        # Remove words with length <= 1 from word count
        for word in purge:
            del word_count[word]
            del tokenized_words[word]

        # Find most common pair and add it to vocabulary
        new_token = "".join(pairs.most_common(1)[0][0])
        vocab.append(new_token)

        # BEGIN[ChatGPT][https://chatgpt.com/]"How to optimize naive BPE implementation? + Python code"
        # Update word count and tokenized words
        contains_new_token = lambda t, n: any(i + j == n for i, j in pairwise(t))
        for word in word_count:
            if contains_new_token(tokenized_words[word], new_token):
                tokenized_words[word] = spacelessBPETokenize(word, vocab)
        # END[ChatGPT]

        # Write top 5 pairs to output file
        if iter in [0, 1, 10, 100, 500]:
            outfile.write(f"iter {iter}: {pairs.most_common(5)}\n")

    # Check if vocabulary size is correct
    assert (
        len(vocab) == max_vocabulary
    ), f"Vocabulary size is {len(vocab)}, expected {max_vocabulary}"
    assert len(set(vocab)) == len(
        vocab
    ), f"Vocabulary has duplicates: {[i for i in vocab if vocab.count(i) > 1]}"

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
        outfile.write(f"{result}\n")
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
    outfile.write(f"{final_vocabulary}\n\n")

    # Tokenize and print first 5 and last doc in input data
    print("Tokenizing first 5 and last doc in input data using SLBPE...")
    outfile.write("Tokenized data:\n")
    for s in test_data:
        result = spacelessBPETokenize(s, final_vocabulary)
        outfile.write(f"{result}\n")

    # Close output file
    print("Closing output file...")
    outfile.close()

    return None


if __name__ == "__main__":
    main()
