#!/usr/bin/env python3

import os
import sys
import argparse

from typing import List


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


def getFeaturesForTarget(tokens, targetI, wordToIndex):
    # input: tokens: a list of tokens in a sentence,
    #        targetI: index for the target token
    #        wordToIndex: dict mapping ‘word’ to an index in the feature list.
    # output: list (or np.array) of k feature values for the given target

    # <FILL IN>
    featureVector = None
    return featureVector


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

    outfile.write("Checkpoint 2.1:\n")
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
