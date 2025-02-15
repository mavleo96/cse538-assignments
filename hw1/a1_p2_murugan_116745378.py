#!/usr/bin/env python3

import os
import sys
import argparse
import numpy as np

from typing import List, Dict
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def getConllTags(filename: str) -> List[List]:
    # input: filename for a conll style parts of speech tagged file
    # output: a list of list of tuples [sent]. representing [[[word1, tag], [word2, tag2]]

    wordTagsPerSent = [[]]
    sentNum = 0
    with open(filename, encoding="utf8") as f:
        for wordtag in f:
            wordtag = wordtag.strip()
            if wordtag:  # still reading current sentence
                (word, tag) = wordtag.split("\t")
                wordTagsPerSent[sentNum].append((word, tag))
            else:  # new sentence
                wordTagsPerSent.append([])
                sentNum += 1
    return wordTagsPerSent


def getFeaturesForTarget(
    tokens: List[str], targetI: int, wordToIndex: Dict[str, int]
) -> np.array:
    """Return an array of features for a token in sentence"""
    assert 0 <= targetI < len(tokens), "list index out of range"

    ftarget_ascii = ord(tokens[targetI][0])

    # Feature 1: Captizalied or not
    capital = np.array([int(ord("A") <= ftarget_ascii <= ord("Z"))])

    # Feature 2: First letter
    f_array = np.zeros(257)
    f_array[ftarget_ascii if ftarget_ascii < 256 else 256] = 1

    # Feature 3: Length of token
    length = np.array([len(tokens[targetI])])

    # Feature 4: Previous token
    previous_token = np.zeros(len(wordToIndex))
    if targetI != 0:
        previous_token[wordToIndex[tokens[targetI - 1]]] = 1

    # Feature 5: Current token
    current_token = np.zeros(len(wordToIndex))
    current_token[wordToIndex[tokens[targetI]]] = 1

    # Feature 6: Next token
    next_token = np.zeros(len(wordToIndex))
    if targetI != len(tokens) - 1:
        next_token[wordToIndex[tokens[targetI + 1]]] = 1

    # Concatenate all features into a single array
    feature_vector = np.concatenate(
        [capital, f_array, length, previous_token, current_token, next_token]
    )

    return feature_vector


def trainLogReg(train_data, dev_data, learning_rate, l2_penalty):
    # input: train/dev_data - contain the features and labels for train/dev splits
    # input: learning_rate, l2_penalty - hyperparameters for model training
    # output: model - the trained pytorch model
    # output: train/dev_losses - a list of train/dev set loss values from each epoch
    # output: train/dev_accuracies - a list of train/dev set accuracy from each epoch
    model = train_losses = train_accuracies = dev_losses = dev_accuracies = None
    return model, train_losses, train_accuracies, dev_losses, dev_accuracies


def gridSearch(train_set, dev_set, learning_rates, l2_penalties):
    # input: learning_rates, l2_penalties - each is a list with hyperparameters to try
    #        train_set - the training set of features and outcomes
    #        dev_set - the dev set of features and outcomes
    # output: model_accuracies - dev set accuracy of the trained model on each hyperparam combination
    #         best_lr, best_l2_penalty - learning rate and L2 penalty combination with highest dev set accuracy
    model_accuracies = best_lr = best_l2_penalty = None
    return model_accuracies, best_lr, best_l2_penalty


def main() -> None:
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="script to run cse538 assignment 1 part 2"
    )
    parser.add_argument("filepath", type=str, help="path to the input file")
    args = parser.parse_args()

    # Read and process input data
    data = getConllTags(args.filepath)

    # Create and open output file
    os.makedirs("results", exist_ok=True)
    outfile = open("results/a1_p2_murugan_116745378_OUTPUT.txt", "w")

    # Create mapping dictionaries
    unique_tokens = set(token for sentence in data for token, _ in sentence)
    unique_postags = set(postag for sentence in data for _, postag in sentence)
    token_index = {token: id for id, token in enumerate(unique_tokens)}
    postag_index = {postag: id for id, postag in enumerate(unique_postags)}

    # Create lexical feature set
    outfile.write("Checkpoint 2.1:\n")
    X = np.array(
        [
            getFeaturesForTarget([i for i, _ in sentence], id, token_index)
            for sentence in data
            for id, _ in enumerate(sentence)
        ]
    )
    y = np.array([postag_index[postag] for sentence in data for _, postag in sentence])

    # Print feature vector sum for first and last 5 rows
    test_data = np.vstack([X[:1, :], X[-5:, :]])
    feature_vector_sum = ",".join(test_data.sum(1).astype(str))
    outfile.write(feature_vector_sum + "\n")

    # Shuffle and split train test data
    X, y = shuffle(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    outfile.write("Checkpoint 2.2:\n")
    outfile.write("Checkpoint 2.3:\n")
    outfile.write("Checkpoint 2.4:\n")

    # Close output file
    outfile.close()

    return None


if __name__ == "__main__":
    main()
else:
    sys.exit(0)
