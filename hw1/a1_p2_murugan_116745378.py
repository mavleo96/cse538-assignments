#!/usr/bin/env python3

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

np.random.seed(0)
torch.manual_seed(0)


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
) -> np.ndarray:
    """Return an array of features for a token in sentence"""
    assert 0 <= targetI < len(tokens), "list index out of range"

    ftarget_ascii = ord(tokens[targetI][0])
    vocab_size = len(wordToIndex)

    # Feature 1: Captizalied or not
    capital = np.array([int(ord("A") <= ftarget_ascii <= ord("Z"))])

    # Feature 2: First letter
    f_array = np.zeros(257)
    f_array[min(ftarget_ascii, 256)] = 1

    # Feature 3: Length of token
    length = np.array([len(tokens[targetI])])

    # Feature 4: Previous token
    previous_token = np.zeros(vocab_size)
    if targetI != 0 and tokens[targetI - 1] in wordToIndex:
        previous_token[wordToIndex[tokens[targetI - 1]]] = 1

    # Feature 5: Current token
    current_token = np.zeros(vocab_size)
    if tokens[targetI] in wordToIndex:
        current_token[wordToIndex[tokens[targetI]]] = 1

    # Feature 6: Next token
    next_token = np.zeros(vocab_size)
    if targetI != len(tokens) - 1 and tokens[targetI + 1] in wordToIndex:
        next_token[wordToIndex[tokens[targetI + 1]]] = 1

    # Concatenate all features into a single array
    feature_vector = np.concatenate(
        [capital, f_array, length, previous_token, current_token, next_token]
    )

    return feature_vector


class MulticlassLogisticRegression(nn.Module):
    """Multiclass Logistic Regression Model"""

    def __init__(self, dim: int, nclass: int) -> None:
        """Initialize the model"""
        super().__init__()
        self.linear = nn.Linear(dim, nclass, dtype=torch.float32)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model"""
        x = self.linear(x)
        x = self.log_softmax(x)

        return x


def trainLogReg(
    train_data: TensorDataset,
    dev_data: TensorDataset,
    learning_rate: float,
    l2_penalty: float,
) -> Tuple[nn.Module, List, List, List, List]:
    """Train a multiclass logistic regression model"""
    assert (
        train_data.tensors[0].shape[1] == dev_data.tensors[0].shape[1]
    ), "train_data and dev_data shape don't match on axis=1"

    feature_count = train_data.tensors[0].shape[1]
    num_class = train_data.tensors[1].max().item() + 1

    train_dataloader = DataLoader(
        train_data, batch_size=len(train_data) // 100, shuffle=True
    )

    # Initialize model, loss function and optimizer
    model = MulticlassLogisticRegression(feature_count, num_class)
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, weight_decay=l2_penalty
    )

    train_losses, dev_losses = [], []
    train_accuracies, dev_accuracies = [], []

    # Training Loop
    for _ in range(200):
        for batch_X, batch_y in train_dataloader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            # Evaluate on train and dev data
            train_logprob_pred = model(train_data.tensors[0])
            dev_logprob_pred = model(dev_data.tensors[0])

            train_y_pred = train_logprob_pred.argmax(1).cpu().numpy()
            dev_y_pred = dev_logprob_pred.argmax(1).cpu().numpy()

            # Calculate & save loss and accuracy
            train_losses.append(
                loss_fn(train_logprob_pred, train_data.tensors[1]).item()
            )
            dev_losses.append(loss_fn(dev_logprob_pred, dev_data.tensors[1]).item())

            train_accuracies.append(
                accuracy_score(train_data.tensors[1].cpu().numpy(), train_y_pred)
            )
            dev_accuracies.append(
                accuracy_score(dev_data.tensors[1].cpu().numpy(), dev_y_pred)
            )

    return model, train_losses, train_accuracies, dev_losses, dev_accuracies


def gridSearch(train_set, dev_set, learning_rates, l2_penalties):
    # input: learning_rates, l2_penalties - each is a list with hyperparameters to try
    #        train_set - the training set of features and outcomes
    #        dev_set - the dev set of features and outcomes
    # output: model_accuracies - dev set accuracy of the trained model on each hyperparam combination
    #         best_lr, best_l2_penalty - learning rate and L2 penalty combination with highest dev set accuracy
    model_accuracies = best_lr = best_l2_penalty = None
    return model_accuracies, best_lr, best_l2_penalty


def main():
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
    unique_tokens = {t for s in data for t, _ in s}
    unique_postags = {p for s in data for _, p in s}
    token_index = {t: i for i, t in enumerate(unique_tokens)}
    postag_index = {p: i for i, p in enumerate(unique_postags)}

    # Create lexical feature set
    outfile.write("Checkpoint 2.1:\n")
    X = np.array(
        [
            getFeaturesForTarget([t for t, _ in s], i, token_index)
            for s in data
            for i, _ in enumerate(s)
        ]
    )
    y = np.array([postag_index[p] for s in data for _, p in s])

    # Print feature vector sum for first and last 5 rows
    test_data = np.vstack([X[:1, :], X[-5:, :]])
    feature_vector_sum = ",".join(test_data.sum(1).astype(str))
    outfile.write(feature_vector_sum + "\n")

    # Split and scale train test data
    X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.3)
    scaler = np.maximum(X_train.max(0, keepdims=True), 1)
    X_train /= scaler
    X_dev /= scaler

    # Create tensor objects
    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.long)
    Xd = torch.tensor(X_dev, dtype=torch.float32)
    yd = torch.tensor(y_dev, dtype=torch.long)

    train_dataset, dev_dataset = TensorDataset(Xt, yt), TensorDataset(Xd, yd)

    # Train logistic regression
    outfile.write("Checkpoint 2.2:\n")

    _, train_losses, train_accuracies, dev_losses, dev_accuracies = trainLogReg(
        train_dataset, dev_dataset, 0.01, 0.01
    )

    # Create a figure with two subplots
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Plot loss on left y-axis
    ax1.plot(train_losses, label="Train Loss", color="blue", linestyle="-")
    ax1.plot(dev_losses, label="Dev Loss", color="red", linestyle="--")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper left")
    ax1.grid()

    # Create a second y-axis for accuracy
    ax2 = ax1.twinx()
    ax2.plot(train_accuracies, label="Train Accuracy", color="green", linestyle="-")
    ax2.plot(dev_accuracies, label="Dev Accuracy", color="orange", linestyle="--")
    ax2.set_ylabel("Accuracy")
    ax2.legend(loc="upper right")

    # Save the plot
    plt.title("Training & Dev Loss and Accuracy")
    plt.savefig("results/training_plot.png", dpi=300, bbox_inches="tight")

    outfile.write("Checkpoint 2.3:\n")
    outfile.write("Checkpoint 2.4:\n")

    # Close output file
    outfile.close()

    return None


if __name__ == "__main__":
    main()
else:
    sys.exit(0)
