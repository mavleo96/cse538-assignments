#!/usr/bin/env python3

import os
import argparse
import csv
import torch
import re
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm
from typing import List, Tuple
from a2_p1_murugan_116745378 import init_tokenizer, get_perplexity


np.random.seed(0)
torch.manual_seed(0)

# ==========================
#     RecurrentLM Class
# ==========================


class RecurrentLM(nn.Module):
    """RNN based language model"""

    def __init__(self, vocab_size: int, embed_dim: int, rnn_hidden_dim: int):
        """Initialize the model"""
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, rnn_hidden_dim, batch_first=True)
        self.layer_norm = nn.LayerNorm(rnn_hidden_dim)
        self.classifier = nn.Linear(rnn_hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model"""
        x = self.embedding(x)
        x, hidden_state = self.gru(x)
        x = self.layer_norm(x)
        logits = self.classifier(x)

        return logits, hidden_state

    # def stepwise_forward(self, x, prev_hidden_state):
    # input: x: tensor of shape (seq_len)
    #       hidden_state: hidden state of GRU after processing x (single token)
    # <FILL IN at Part 2.4>

    # return logits, hidden_state


# ==========================
#     Training Functions
# ==========================


def trainLM(
    model: nn.Module,
    data: DataLoader,
    pad_token_id: int,
    learning_rate: float,
    device: str,
) -> Tuple[nn.Module, List[float]]:
    """Train the language model"""
    num_epochs = 15

    # Initialize optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    model.to(device)
    loss_fn.to(device)

    # Loop through epochs
    losses = []
    model.train()
    for _ in tqdm(range(num_epochs), desc="Epochs"):
        epoch_loss = 0

        # Loop through batches
        for batch in data:
            # Get batch data and move to device
            X, y = batch
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(X)
            loss = loss_fn(logits.view(-1, logits.shape[-1]), y.view(-1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()  # Accumulate loss

        losses.append(epoch_loss / len(data))  # Append average loss for epoch

    return model, losses


# ==========================
#     Helper Functions
# ==========================


def chunk_tokens(
    tokens: List[int],
    start_token_id: int,
    end_token_id: int,
    pad_token_id: int,
    chunk_len: int = 128,
) -> torch.Tensor:
    """Chunk tokens into chunks of length chunk_len"""
    u_chunk_len = chunk_len - 2  # Remove start and end tokens from chunk length
    assert u_chunk_len > 0, "Chunk length must be greater than 2"

    # Pad tokens if necessary
    pad_count = (
        (u_chunk_len - len(tokens) % u_chunk_len)
        if len(tokens) % u_chunk_len != 0
        else 0
    )
    tokens += [pad_token_id] * pad_count

    # Chunk tokens by reshaping
    chunked_tokens = torch.tensor(tokens).reshape(-1, u_chunk_len)
    n_chunks = chunked_tokens.shape[0]

    # Concatenate BOS and EOS tokens
    bos_tensor = torch.full((n_chunks, 1), start_token_id)
    eos_tensor = torch.full((n_chunks, 1), end_token_id)
    chunks = torch.cat((bos_tensor, chunked_tokens, eos_tensor), dim=1)

    return chunks


def process_data(
    data: List[List[str]], tokenizer: PreTrainedTokenizerFast, chunk_len: int
) -> torch.Tensor:
    """Process, tokenize, and chunk the data"""
    PATTERN = r"\n\[[\x20-\x7f]+\]"

    # Remove section markers and tokenize the data
    data = [re.sub(PATTERN, "", row[2]) for row in data]
    tokenized_data = [tokenizer.encode(row) for row in data]

    # Chunk the tokens
    chunked_data = [
        chunk_tokens(
            tokens,
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
            tokenizer.pad_token_id,
            chunk_len,
        )
        for tokens in tokenized_data
    ]

    # Concatenate the chunks
    processed_data = torch.cat(chunked_data, dim=0)

    return processed_data


# ==========================
#     Main Function
# ==========================


def main() -> None:
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="script to run cse538 assignment 2 part 2"
    )
    parser.add_argument("filepath", type=str, help="path to the input file")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--chunk_len", type=int, default=64, help="chunk length")
    parser.add_argument("--embed_dim", type=int, default=64, help="embedding dimension")
    parser.add_argument(
        "--rnn_hidden_dim", type=int, default=1024, help="rnn hidden dimension"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0007, help="learning rate"
    )
    parser.add_argument("--save_model", action="store_true", help="save the model")

    args = parser.parse_args()
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Read and process input data
    with open(args.filepath, "r", newline="") as f:
        reader = csv.reader(f)
        data = list(reader)[1:-5]

    # Create and open output file
    print("Creating output file...")
    os.makedirs("results", exist_ok=True)
    outfile = open("results/a2_p2_murugan_116745378_OUTPUT.txt", "w")
    # TODO: check if pdf output can be directly created instead

    # Preparing the dataset
    outfile.write("Checkpoint 2.1:\n")

    # Initialize GPT2 tokenizer
    print("Initializing GPT2 tokenizer...")
    tokenizer = init_tokenizer()

    # Process the data
    processed_data = process_data(data, tokenizer, chunk_len=args.chunk_len)
    X = processed_data[:, :-1]
    y = processed_data[:, 1:]
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, drop_last=True)

    # Output chunked tensor for "Enchanted (Taylor's Version)"
    test_data = [
        row
        for row in data
        if row[0] == "Enchanted (Taylor's Version)"
        and row[1] == "Speak Now (Taylor's Version)"
    ]
    processed_test_data = process_data(test_data, tokenizer, chunk_len=args.chunk_len)
    outfile.write(
        f'Chunked tensor for "Enchanted (Taylor\'s Version)":\n{processed_test_data}\n\n'
    )

    # Print shape of logits and hidden state
    outfile.write("Checkpoint 2.2:\n")

    print("Writing logits and hidden state shapes to output file...")
    logits_shape = f"(batch_size, chunk_len - 1, vocab_size) -> ({args.batch_size}, {args.chunk_len - 1}, {len(tokenizer.vocab)})"
    hidden_state_shape = f"(num_layers, batch_size, rnn_hidden_dim) -> (1, {args.batch_size}, {args.rnn_hidden_dim})"
    outfile.write(
        f"Logits shape: {logits_shape}\nHidden state shape: {hidden_state_shape}\n\n"
    )

    outfile.write("Checkpoint 2.3:\n")

    # Initialize and train RecurrentLM
    print("Initializing model...")
    model = RecurrentLM(len(tokenizer.vocab), args.embed_dim, args.rnn_hidden_dim)

    print("Training model...")
    model, losses = trainLM(
        model, dataloader, tokenizer.pad_token_id, args.learning_rate, device
    )
    if args.save_model:
        print("Saving model weights to results/model.pt...")
        torch.save(model.state_dict(), "results/model.pt")

    print("Saving losses plot...")
    plt.plot(losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Losses")
    plt.savefig("results/training_plot.png", dpi=300, bbox_inches="tight")
    outfile.write("Losses plot saved to results/training_plot.png\n\n")

    # Evaluate model perplexity for test data
    print("Evaluating model perplexity for test data...")
    outfile.write("Perplexity for test data:\n")
    test_data = [
        "And you gotta live with the bad blood now",
        "Sit quiet by my side in the shade",
        "And I'm not even sorry, nights are so starry",
        "You make me crazier, crazier, crazier, oh",
        "When time stood still and I had you",
    ]
    for i in test_data:
        tokens = tokenizer.encode(i)
        input_tensor = torch.tensor(
            [tokenizer.bos_token_id] + tokens[:-1], device=device
        )
        output_tensor = torch.tensor(tokens, device=device)

        model.eval()
        with torch.no_grad():
            logits, _ = model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)
            probabilities = (
                torch.gather(probabilities, 1, output_tensor.unsqueeze(1))
                .squeeze(1)
                .tolist()
            )

        perplexity = get_perplexity(probabilities)
        outfile.write(f"'{i}': {perplexity:.2f}\n")

    # Close output file
    print("Closing output file...")
    outfile.close()

    return None


if __name__ == "__main__":
    main()
