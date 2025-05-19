from collections import defaultdict
import re
from typing import List, Dict, Tuple, Set
import json


class BytePairEncoding:
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = {}
        self.reverse_vocab = {}

    def get_stats(self, words: List[List[str]]) -> Dict[Tuple[str, str], int]:
        """Count frequency of adjacent pairs."""
        pairs = defaultdict(int)
        for word in words:
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += 1
        return pairs

    def merge_pair(
        self, pair: Tuple[str, str], words: List[List[str]]
    ) -> List[List[str]]:
        """Merge all occurrences of the pair in the words."""
        new_words = []
        for word in words:
            i = 0
            new_word = []
            while i < len(word):
                if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                    new_word.append(pair[0] + pair[1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_words.append(new_word)
        return new_words

    def train(self, text: str):
        """Train the BPE tokenizer on the given text."""
        # Initialize vocabulary with characters
        words = [list(word) for word in text.split()]
        vocab = set()
        for word in words:
            vocab.update(word)

        # Initialize vocabulary
        self.vocab = {i: char for i, char in enumerate(vocab)}
        self.reverse_vocab = {char: i for i, char in self.vocab.items()}

        # Perform merges
        num_merges = self.vocab_size - len(vocab)
        for i in range(num_merges):
            pairs = self.get_stats(words)
            if not pairs:
                break

            best_pair = max(pairs.items(), key=lambda x: x[1])[0]
            words = self.merge_pair(best_pair, words)
            self.merges[best_pair] = i + len(vocab)

            # Update vocabulary
            merged_token = best_pair[0] + best_pair[1]
            self.vocab[i + len(vocab)] = merged_token
            self.reverse_vocab[merged_token] = i + len(vocab)

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using learned BPE merges."""
        words = text.split()
        tokenized_words = []

        for word in words:
            chars = list(word)
            i = 0
            while i < len(chars) - 1:
                pair = (chars[i], chars[i + 1])
                if pair in self.merges:
                    chars[i : i + 2] = [pair[0] + pair[1]]
                else:
                    i += 1
            tokenized_words.extend(chars)

        return tokenized_words


# TODO: Problem here! non-unique tokens
class WordPiece:
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.reverse_vocab = {}

    def get_word_frequencies(self, text: str) -> Dict[str, int]:
        """Get frequency of each word in the text."""
        words = text.split()
        return defaultdict(int, {word: words.count(word) for word in set(words)})

    def get_subword_frequencies(self, words: Dict[str, int]) -> Dict[str, int]:
        """Get frequency of each subword in the words."""
        subword_freq = defaultdict(int)
        for word, freq in words.items():
            for i in range(len(word)):
                for j in range(i + 1, len(word) + 1):
                    subword = word[i:j]
                    subword_freq[subword] += freq
        return subword_freq

    def train(self, text: str):
        """Train the WordPiece tokenizer on the given text."""
        # Initialize vocabulary with characters
        chars = set(text)
        self.vocab = {i: char for i, char in enumerate(chars)}
        self.reverse_vocab = {char: i for i, char in self.vocab.items()}

        # Get word frequencies
        word_freq = self.get_word_frequencies(text)

        # Iteratively add subwords
        num_iterations = self.vocab_size - len(chars)
        for _ in range(num_iterations):
            subword_freq = self.get_subword_frequencies(word_freq)
            if not subword_freq:
                break

            # Add most frequent subword to vocabulary
            best_subword = max(subword_freq.items(), key=lambda x: x[1])[0]
            self.vocab[len(self.vocab)] = best_subword
            self.reverse_vocab[best_subword] = len(self.vocab) - 1

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using learned WordPiece vocabulary."""
        words = text.split()
        tokenized_words = []

        for word in words:
            start = 0
            subwords = []
            while start < len(word):
                end = len(word)
                cur_substr = None

                while start < end:
                    substr = word[start:end]
                    if substr in self.reverse_vocab:
                        cur_substr = substr
                        break
                    end -= 1

                if cur_substr is None:
                    # Handle unknown characters
                    cur_substr = word[start : start + 1]

                subwords.append(cur_substr)
                start = end

            tokenized_words.extend(subwords)

        return tokenized_words


if __name__ == "__main__":
    import sys

    sys.path.append("../")
    with open("practice/malcolm-gladwell.txt", "r") as f:
        text = f.read()

    # Test BPE
    print("Testing Byte Pair Encoding:")
    bpe = BytePairEncoding(vocab_size=100)
    bpe.train(text)
    bpe_tokens = bpe.tokenize(text)
    print("BPE Tokens:", bpe_tokens)  # Print first 20 tokens
    print("Unique BPE Tokens:", len(set(bpe_tokens)))

    # Test WordPiece
    print("\nTesting WordPiece:")
    wp = WordPiece(vocab_size=100)
    wp.train(text)
    wp_tokens = wp.tokenize(text)
    print("WordPiece Tokens:", wp_tokens)  # Print first 20 tokens
    print("Unique WordPiece Tokens:", len(set(wp_tokens)))
