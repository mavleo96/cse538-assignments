#!/usr/bin/env python3

import os
import argparse
import csv
import torch
import re
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from itertools import batched
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

    def stepwise_forward(
        self, x: torch.Tensor, prev_hidden_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stepwise forward pass through the model"""
        x = self.embedding(x)
        x, hidden_state = self.gru(x, prev_hidden_state)
        x = self.layer_norm(x)
        logits = self.classifier(x)

        return logits, hidden_state


# ==========================
#  Training and Generation
# ==========================


def trainLM(
    model: nn.Module,
    data: DataLoader,
    pad_token_id: int,
    learning_rate: float,
    device: str,
) -> List[float]:
    """Train the language model"""
    num_epochs = 15

    model.to(device)  # Move model to device

    # Initialize optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)

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
            loss = loss_fn(
                logits.transpose(1, 2), y
            )  # loss_fn expects ((b, c, ..), (b, ..))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()  # Accumulate loss

        losses.append(epoch_loss / len(data))  # Append average loss for epoch

    return losses


def generate(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerFast,
    start_phrase: str,
    max_len: int,
    device: str,
    sample: bool = False,
) -> List[int]:
    """Generate a sequence of tokens from the model"""
    # shape should be (b, s) to be compatible with layer norm in the model
    start_tokens = torch.tensor(
        tokenizer.encode(start_phrase), device=device, dtype=torch.long
    ).unsqueeze(0)
    generated_tokens = []

    model.eval()
    with torch.no_grad():
        logits, hidden_state = model(start_tokens)
        if sample:
            next_token = torch.distributions.Categorical(
                logits=logits[:, -1:, :]
            ).sample()
        else:
            next_token = torch.argmax(logits, dim=2)[:, -1:]
        generated_tokens.append(next_token.item())

        while (len(generated_tokens) < max_len) and (
            generated_tokens[-1] not in [tokenizer.eos_token_id, tokenizer.pad_token_id]
        ):
            logits, hidden_state = model.stepwise_forward(next_token, hidden_state)
            if sample:
                next_token = torch.distributions.Categorical(
                    logits=logits[:, -1:, :]
                ).sample()
            else:
                next_token = torch.argmax(logits, dim=2)[:, -1:]
            generated_tokens.append(next_token.item())

    return generated_tokens


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
    assert len(tokens) > 0, "Tokens must not be empty"

    # Chunk tokens into batches
    chunked_list = []
    for batch in batched(tokens, u_chunk_len):
        batch = [start_token_id, *batch, end_token_id]
        # Pad with pad_token_id if necessary
        if (pad_count := (chunk_len - len(batch))) > 0:
            batch += [pad_token_id] * pad_count
        assert len(batch) == chunk_len, "Batch length must be equal to chunk length"
        chunked_list.append(batch)

    # Convert to tensor
    chunks = torch.tensor(chunked_list, dtype=torch.long)
    n_chunks = chunks.shape[0]

    # Check chunk properties
    assert n_chunks == np.ceil(len(tokens) / u_chunk_len)
    assert set(chunks[:, 0].unique()) == {start_token_id}
    assert set(chunks[:, -1].unique()) == {end_token_id, pad_token_id}
    return chunks


def process_data(
    data: List[List[str]], tokenizer: PreTrainedTokenizerFast, chunk_len: int
) -> torch.Tensor:
    """Process, tokenize, and chunk the data"""
    assert len(data) > 0, "Data must not be empty"
    assert all(len(row) == 3 for row in data), "Data must have 3 columns"

    # Remove section markers and tokenize the data
    pattern = r"\n\[[\x20-\x7f]+\]"
    data = [re.sub(pattern, "", row[2]) for row in data]
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
#     Observations
# ==========================


OBSERVATIONS1 = """Observations:
We observe the the perplexity score for RNN model is 10 to 20 times lower than trigram model.
This is expected because the RNN model is able to capture the context of the data better than the trigram model by using the hidden states.
"""

OBSERVATIONS2 = """Observations:
The content generated by sampling over all logits is more unique than by argmaxing over the logits.
This is because argmaxing over the logits is more likely to generate the most common tokens, which tend to mimic the training data more.
Sampling over all logits allows the model to generate more unique tokens.
"""


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

    # Preparing the dataset
    outfile.write("Checkpoint 2.1:\n")

    # Initialize GPT2 tokenizer
    print("Initializing GPT2 tokenizer...")
    tokenizer = init_tokenizer()

    # Process the data
    processed_data = process_data(data, tokenizer, chunk_len=args.chunk_len)
    X, y = processed_data[:, :-1], processed_data[:, 1:]
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
        f'Chunked tensor for "Enchanted (Taylor\'s Version)"\n{processed_test_data}\n'
    )
    outfile.write("\n")

    # Print shape of logits and hidden state
    outfile.write("Checkpoint 2.2:\n")

    print("Writing logits and hidden state shapes to output file...")
    logits_shape = f"(batch_size, chunk_len - 1, vocab_size) = ({args.batch_size}, {args.chunk_len - 1}, {len(tokenizer.vocab)})"
    hidden_state_shape = f"(num_layers, batch_size, rnn_hidden_dim) = (1, {args.batch_size}, {args.rnn_hidden_dim})"
    outfile.write(
        f"Logits shape: {logits_shape}\nHidden state shape: {hidden_state_shape}\n"
    )
    outfile.write("\n")

    # Initialize and train RecurrentLM
    outfile.write("Checkpoint 2.3:\n")
    print("Initializing model...")
    model = RecurrentLM(len(tokenizer.vocab), args.embed_dim, args.rnn_hidden_dim)

    print("Training model...")
    losses = trainLM(
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
    for seq in test_data:
        tokens = [tokenizer.bos_token_id] + tokenizer.encode(seq)
        token_tensor = torch.tensor(tokens, device=device, dtype=torch.long)
        input_tensor, output_tensor = token_tensor[:-1], token_tensor[1:]

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
        outfile.write(f"'{seq}': {perplexity:.2f}\n")
    outfile.write("\n")

    # Write observations on perplexity
    outfile.write(OBSERVATIONS1)
    outfile.write("\n")

    # Stepwise forward pass
    print("Generating text by argmaxing over the logits...")
    outfile.write("Checkpoint 2.4:\n")
    test_data = [
        "<s>Are we",
        "<s>Like we're made of starlight, starlight",
        "<s>I hate Calvin Harris and he is",
    ]
    for i, seq in enumerate(test_data):
        outfile.write(f"====== Song {i+1} =====\n")
        generated_tokens = generate(model, tokenizer, seq, 64, device)
        text = tokenizer.decode(generated_tokens)
        outfile.write(f"{seq}{text}\n")
        outfile.write(f"===================\n")
        outfile.write("\n")

    # Stepwise forward pass with sampling
    print("Generating text by sampling over all logits...")
    outfile.write("Checkpoint 2.5: Extra Credit\n")
    for i, seq in enumerate(test_data):
        outfile.write(f"====== Song {i+1} =====\n")
        generated_tokens = generate(model, tokenizer, seq, 64, device, sample=True)
        text = tokenizer.decode(generated_tokens)
        outfile.write(f"{seq}{text}\n")
        outfile.write(f"===================\n")
        outfile.write("\n")

    # Write observations on generated text
    outfile.write(OBSERVATIONS2)

    # Close output file
    print("Closing output file...")
    outfile.close()

    return None


if __name__ == "__main__":
    main()
