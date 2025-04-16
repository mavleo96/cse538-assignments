#! /usr/bin/env python3

import argparse
import os
from typing import Any, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from einops import rearrange
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ==========================
#     Training Functions
# ==========================


def get_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    indices: Union[List[int], None],
    device: torch.device,
    desc: str = "Validation",
) -> Tuple[List[int], List[torch.Tensor]]:
    model.eval()
    label_pred, logits = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            inputs = batch.to(device)
            outputs = model(inputs).logits
            if indices is not None:
                outputs = outputs[:, -1, indices]
            preds = outputs.argmax(dim=-1)
            label_pred.extend(preds.cpu().tolist())
            logits.extend(outputs.cpu().tolist())
    return label_pred, logits


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    save_model: bool,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    train_losses, dev_losses = [], []
    train_accuracies, dev_accuracies = [], []
    progress_bar = tqdm(range(epochs), desc="Training", leave=True)
    for epoch in progress_bar:
        model.train()
        running_loss, running_correct, running_total = 0.0, 0, 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]", leave=False)
        for batch in train_bar:
            # Get batch
            inputs = batch.to(device)
            X, y = inputs[:, :-1], inputs[:, 1:]

            # Forward pass
            scaler = torch.amp.GradScaler()
            with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
                logits = model(X).logits
                loss = loss_fn(rearrange(logits, "b s c -> b c s"), y)

            # Backward pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Update metrics
            running_loss += loss.item() * inputs.size(0)
            preds = logits.argmax(dim=-1)
            running_correct += (preds == y).sum().item()
            running_total += inputs.size(0)
            train_bar.set_postfix(loss=loss.item(), acc=running_correct / running_total)

        train_losses.append(running_loss / running_total)
        train_accuracies.append(running_correct / running_total)

        # Validation loop
        model.eval()
        running_loss, running_correct, running_total = 0.0, 0, 0
        with torch.no_grad():
            val_bar = tqdm(
                val_loader, desc=f"Epoch {epoch + 1} [Validation]", leave=False
            )
            for batch in val_bar:
                # Get batch
                inputs = batch.to(device)
                X, y = inputs[:, :-1], inputs[:, 1:]

                # Forward pass
                with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
                    logits = model(X).logits
                    loss = loss_fn(rearrange(logits, "b s c -> b c s"), y)

                # Update metrics
                preds = logits.argmax(dim=-1)
                running_loss += loss.item() * inputs.size(0)
                running_correct += (preds == y).sum().item()
                running_total += inputs.size(0)
                val_bar.set_postfix(
                    loss=loss.item(), acc=running_correct / running_total
                )
        dev_losses.append(running_loss / running_total)
        dev_accuracies.append(running_correct / running_total)

    progress_bar.close()

    if save_model:
        print("Saving model...")
        torch.save(model.state_dict(), "results/a3_p1_murugan_116745378_model.pt")

    return train_losses, dev_losses, train_accuracies, dev_accuracies


# ==========================
#     Helper Functions
# ==========================


def boolq2tensor(
    x: Dict[str, Any], tokenizer: GPT2Tokenizer, append_answer: bool = False
) -> torch.Tensor:
    if append_answer:
        text = f"{x['passage']}.\n{x['question']}?\n{'yes' if x['answer'] else 'no'}"
    else:
        text = f"{x['passage']}.\n{x['question']}?"
    return tokenizer.encode(text, return_tensors="pt")


def tensorlist2padded(
    tensorlist: List[torch.Tensor], length: int, pad_token_id: int, pad_strategy: str
) -> torch.Tensor:
    assert len(tensorlist) > 0
    assert all(x.ndim == 2 for x in tensorlist)
    assert all(x.shape[0] == 1 for x in tensorlist)
    assert length > 0
    assert pad_token_id is not None
    assert pad_strategy in {"left", "right"}

    for i, x in enumerate(tensorlist):
        if x.shape[1] < length:
            if pad_strategy == "left":
                tensorlist[i] = torch.cat(
                    [torch.full((1, length - x.shape[1]), pad_token_id), x], dim=1
                )
            else:
                tensorlist[i] = torch.cat(
                    [x, torch.full((1, length - x.shape[1]), pad_token_id)], dim=1
                )
        elif x.shape[1] > length:
            if pad_strategy == "left":
                tensorlist[i] = x[:, -length:]
            else:
                tensorlist[i] = x[:, :length]
        else:
            assert x.shape[1] == length

    concat_tensor = torch.cat(tensorlist, dim=0)
    return concat_tensor


def get_metric_str(labels: List[int], label_pred: List[int]) -> str:
    acc = accuracy_score(labels, label_pred)
    f1 = f1_score(labels, label_pred, zero_division=0, average="macro")
    prec_yes = precision_score(labels, label_pred, pos_label=1, zero_division=0)
    recall_yes = recall_score(labels, label_pred, pos_label=1, zero_division=0)
    prec_no = precision_score(labels, label_pred, pos_label=0, zero_division=0)
    recall_no = recall_score(labels, label_pred, pos_label=0, zero_division=0)
    f1_yes = f1_score(labels, label_pred, pos_label=1, zero_division=0)
    f1_no = f1_score(labels, label_pred, pos_label=0, zero_division=0)

    return f"""Overall: acc: {acc:.3f}, f1: {f1:.3f}
    Yes: prec: {prec_yes:.3f}, recall: {recall_yes:.3f}, f1: {f1_yes:.3f}
     No: prec: {prec_no:.3f}, recall: {recall_no:.3f}, f1: {f1_no:.3f}"""


def plot_loss_and_accuracy(
    train_losses: List[float],
    dev_losses: List[float],
    train_accuracies: List[float],
    dev_accuracies: List[float],
    filename: str,
) -> None:
    """Plot loss and accuracy for training and dev data"""
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


# ==========================
#     Main Function
# ==========================


def main() -> None:
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="script to run cse538 assignment 3 part 1"
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--save_model", action="store_true", default=False)
    args = parser.parse_args()
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Create and open output file
    print("Creating output file...")
    os.makedirs("results", exist_ok=True)
    outfile = open("results/a3_p1_murugan_116745378_OUTPUT.txt", "w")

    # Initialize tokenizer
    print("Initializing tokenizer and distilgpt2 model...")
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    model = GPT2LMHeadModel.from_pretrained("distilgpt2").to(device)

    # Load datasets
    print("Loading BoolQ dataset...")
    boolq_dataset = load_dataset("google/boolq")
    val_dataset = [boolq2tensor(x, tokenizer) for x in boolq_dataset["validation"]]
    train_dataset = [
        boolq2tensor(x, tokenizer, append_answer=True) for x in boolq_dataset["train"]
    ]
    val_tensor = tensorlist2padded(
        val_dataset, args.context_length, tokenizer.unk_token_id, "left"
    )
    train_tensor = tensorlist2padded(
        train_dataset, args.context_length, tokenizer.unk_token_id, "left"
    )
    val_loader = DataLoader(val_tensor, batch_size=args.batch_size, shuffle=False)
    train_loader = DataLoader(train_tensor, batch_size=args.batch_size, shuffle=True)
    val_labels = [1 if x["answer"] else 0 for x in boolq_dataset["validation"]]
    no_token_id, yes_token_id = tokenizer.encode("no")[0], tokenizer.encode("yes")[0]

    # Zero-shot accuracy of distilgpt2 on BoolQ
    outfile.write("Checkpoint 1.1:\n")
    print("Evaluating Zero-shot accuracy of distilgpt2 on BoolQ...")
    label_pred, _ = get_predictions(
        model, val_loader, [no_token_id, yes_token_id], device, "Zero-shot DistilGPT2"
    )
    outfile.write(get_metric_str(val_labels, label_pred) + "\n\n")

    # Finetune distilgpt2 on BoolQ
    outfile.write("Checkpoint 1.2:\n")
    print("Finetuning distilgpt2 on BoolQ...")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.unk_token_id)

    train_losses, dev_losses, train_accuracies, dev_accuracies = train_model(
        model,
        train_loader,
        val_loader,
        loss_fn,
        optimizer,
        device,
        args.epochs,
        args.save_model,
    )
    outfile.write(
        "Training plot saved to results/a3_p1_murugan_116745378_loss_accuracy.png\n\n"
    )
    plot_loss_and_accuracy(
        train_losses,
        dev_losses,
        train_accuracies,
        dev_accuracies,
        "results/a3_p1_murugan_116745378_loss_accuracy.png",
    )

    # Zero-shot accuracy of finetuned distilgpt2 on BoolQ
    outfile.write("Checkpoint 1.3:\n")
    print("Evaluating Zero-shot accuracy of finetuned distilgpt2 on BoolQ...")
    label_pred, _ = get_predictions(
        model,
        val_loader,
        [no_token_id, yes_token_id],
        device,
        "Instruction-tuned DistilGPT2",
    )
    outfile.write(get_metric_str(val_labels, label_pred) + "\n\n")

    # Close output file
    print("Closing output file...")
    outfile.close()

    return None


if __name__ == "__main__":
    main()
