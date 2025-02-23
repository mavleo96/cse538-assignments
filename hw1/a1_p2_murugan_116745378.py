#!/usr/bin/env python3

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Dict, Tuple
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import TensorDataset
from tqdm import tqdm
from tabulate import tabulate
from a1_p1_murugan_116745378 import wordTokenizer

np.random.seed(0)
torch.manual_seed(0)

EPOCHS = 100


# ==========================
#     Utility functions
# ==========================
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


def plot_loss_and_accuracy(
    train_losses: List[float],
    dev_losses: List[float],
    train_accuracies: List[float],
    dev_accuracies: List[float],
    filename: str,
) -> None:
    """Plot loss and accuracy for training and dev data"""
    # BEGIN[Github Copilot][https://github.com/features/copilot]"Code to plot loss and accuracy for training and dev data on the same plot"
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
    print(f"Saving plot to {filename}...")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    # END[Github Copilot]


# ==========================
#     Feature Processing
# ==========================
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

    # Feature 3: Normalized length of token
    length = np.array([min(len(tokens[targetI]), 10) / 10])

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


# ==========================
#     Model Processing
# ==========================
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
    Xt, yt = train_data.tensors
    Xd, yd = dev_data.tensors

    assert (
        Xt.shape[1] == Xd.shape[1]
    ), "train_data and dev_data shape don't match on axis=1"

    feature_count = Xt.shape[1]
    num_class = yt.max().item() + 1

    # Initialize model, loss function and optimizer
    model = MulticlassLogisticRegression(feature_count, num_class)
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, weight_decay=l2_penalty
    )

    train_losses, dev_losses = [], []
    train_accuracies, dev_accuracies = [], []

    # Training Loop
    for _ in tqdm(range(EPOCHS), desc="Training Progress"):
        optimizer.zero_grad()
        loss = loss_fn(model(Xt), yt)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            # Evaluate on train and dev data
            train_logprob_pred = model(Xt)
            dev_logprob_pred = model(Xd)

            train_y_pred = train_logprob_pred.argmax(1).cpu().numpy()
            dev_y_pred = dev_logprob_pred.argmax(1).cpu().numpy()

            # Calculate & save loss and accuracy
            train_losses.append(loss_fn(train_logprob_pred, yt).item())
            dev_losses.append(loss_fn(dev_logprob_pred, yd).item())

            train_accuracies.append(accuracy_score(yt.cpu().numpy(), train_y_pred))
            dev_accuracies.append(accuracy_score(yd.cpu().numpy(), dev_y_pred))

    return model, train_losses, train_accuracies, dev_losses, dev_accuracies


def gridSearch(
    train_set: TensorDataset,
    dev_set: TensorDataset,
    learning_rates: List[float],
    l2_penalties: List[float],
) -> Tuple[np.ndarray, float, float]:
    """Perform grid search to find the best hyperparameters"""

    model_accuracies = np.empty((len(learning_rates), len(l2_penalties)), dtype=float)
    for i, lr in enumerate(learning_rates):
        for j, l2 in enumerate(l2_penalties):
            _, _, _, _, dev_accuracies = trainLogReg(train_set, dev_set, lr, l2)
            model_accuracies[i, j] = dev_accuracies[-1]

    # Find the best learning rate and l2 penalty
    # START[Github Copilot][https://github.com/features/copilot]"Find best hyperparameters"
    argmax = np.unravel_index(model_accuracies.argmax(), model_accuracies.shape)
    best_lr, best_l2_penalty = learning_rates[argmax[0]], l2_penalties[argmax[1]]
    # END[Github Copilot]

    return model_accuracies, best_lr, best_l2_penalty


# ==========================
#        Observations
# ==========================
OBSERVATIONS = """Qualitative Observations:
1. Performance on the test data is below average, likely because 50% of the tokens are out-of-bag vocabulary.
2. The model has learned that the token "the" (not "The") is generally followed by a noun, as it correctly predicted the tags for "barn" and "CS" but not for "horse".
3. It has also learned that tokens with capital letters are generally nouns, as it correctly predicted the tags for "S.B.U.", "CS", and "Sam", despite them being out-of-bag vocabulary.
"""

# ==========================
#       Main Function
# ==========================


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="script to run cse538 assignment 1 part 2"
    )
    parser.add_argument("filepath", type=str, help="path to the input file")
    parser.add_argument("--save_model", action="store_true", help="save the model")
    args = parser.parse_args()

    # Read and process input data
    data = getConllTags(args.filepath)

    # Create and open output file
    print("Creating output file...")
    os.makedirs("results", exist_ok=True)
    outfile = open("results/a1_p2_murugan_116745378_OUTPUT.txt", "w")

    # Create mapping dictionaries
    unique_tokens = {t for s in data for t, _ in s}
    unique_postags = {p for s in data for _, p in s}
    token_index = {t: i for i, t in enumerate(sorted(list(unique_tokens)))}
    postag_index = {p: i for i, p in enumerate(sorted(list(unique_postags)))}

    # Create lexical feature set
    outfile.write("Checkpoint 2.1:\n")
    print("Processing data...")
    X = np.array(
        [
            getFeaturesForTarget([t for t, _ in s], i, token_index)
            for s in data
            for i, _ in enumerate(s)
        ]
    )
    y = np.array([postag_index[p] for s in data for _, p in s])

    # Print feature vector sum for first and last 5 rows
    print("Writing feature vector sum to output file...")
    test_data = np.vstack([X[:1, :], X[-5:, :]])
    feature_vector_sum = test_data.sum(1)
    outfile.write(f"{feature_vector_sum.tolist()}\n\n")

    # Split train test data
    X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.3)

    # Create tensor objects
    print("Creating tensor objects...")
    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.long)
    Xd = torch.tensor(X_dev, dtype=torch.float32)
    yd = torch.tensor(y_dev, dtype=torch.long)

    train_dataset, dev_dataset = TensorDataset(Xt, yt), TensorDataset(Xd, yd)

    # Train logistic regression with lr 0.01 and l2 penalty 0.01
    outfile.write("Checkpoint 2.2:\n")
    print("Training logistic regression model...")
    _, train_losses, train_accuracies, dev_losses, dev_accuracies = trainLogReg(
        train_dataset, dev_dataset, 0.01, 0.01
    )
    plot_loss_and_accuracy(
        train_losses,
        dev_losses,
        train_accuracies,
        dev_accuracies,
        "results/training_plot.png",
    )
    outfile.write("Accuracy plot saved to results/training_plot.png\n\n")

    # Hyperparameter grid search
    outfile.write("Checkpoint 2.3:\n")
    print("Performing grid search...")
    learning_rates = [0.1, 1, 10]
    l2_penalties = [1e-5, 1e-3, 1e-1]
    model_accuracies, best_lr, best_l2_penalty = gridSearch(
        train_dataset, dev_dataset, learning_rates, l2_penalties
    )

    # Print best hyperparameters and model accuracies
    accuracy_table = tabulate(
        model_accuracies.round(3),
        headers=l2_penalties,
        showindex=learning_rates,
        tablefmt="pretty",
    )
    outfile.write(accuracy_table + "\n")
    outfile.write(f"Best hyperparameters: lr={best_lr}, l2_penalty={best_l2_penalty}\n")

    # Train logistic regression with best hyperparameters and plot loss & accuracy
    print("Training logistic regression model with best hyperparameters...")
    (
        best_model,
        best_train_losses,
        best_train_accuracies,
        best_dev_losses,
        best_dev_accuracies,
    ) = trainLogReg(train_dataset, dev_dataset, best_lr, best_l2_penalty)
    plot_loss_and_accuracy(
        best_train_losses,
        best_dev_losses,
        best_train_accuracies,
        best_dev_accuracies,
        "results/best_training_plot.png",
    )
    outfile.write("Accuracy plot saved to results/best_training_plot.png\n\n")

    # Save model
    if args.save_model:
        print("Saving model to results/best_model.pt...")
        torch.save(best_model.state_dict(), "results/best_model.pt")

    # Model inference on test data
    outfile.write("Checkpoint 2.4:\n")
    print("Predicting POS tags for test data...")
    sampleSentences = [
        "The horse raced past the barn fell.",
        "For 3 years, we attended S.B.U. in the CS program.",
        'Did you hear Sam tell me to "chill out" yesterday? #rude',
    ]
    test_data = [wordTokenizer(s) for s in sampleSentences]
    X_test = np.array(
        [
            getFeaturesForTarget(s, i, token_index)
            for s in test_data
            for i, _ in enumerate(s)
        ]
    )
    X_test = torch.tensor(X_test, dtype=torch.float32)

    # Predict POS tags for test data
    test_logprob_pred = best_model(X_test)
    test_y_pred = test_logprob_pred.argmax(1).cpu().numpy()
    inv_postag_index = {v: k for k, v in postag_index.items()}
    test_y_pred = [inv_postag_index[i] for i in test_y_pred]

    # Print predicted POS tags
    i = 0
    for s in test_data:
        predicted_tags = [(t, test_y_pred[i + j]) for j, t in enumerate(s)]
        i += len(s)
        outfile.write(f"{predicted_tags}\n")
    outfile.write("\n")

    # Qualitative observation
    outfile.write(OBSERVATIONS + "\n")

    # Close output file
    print("Closing output file...")
    outfile.close()

    return None


if __name__ == "__main__":
    main()
