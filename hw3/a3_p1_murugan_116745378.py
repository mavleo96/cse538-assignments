#! /usr/bin/env python3

import argparse
import os
from typing import Any, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, GPT2Tokenizer, RobertaTokenizer

# ==========================
#     Data Functions
# ==========================


def process_data_boolq(
    split: str,
    tokenizer: GPT2Tokenizer,
    pad_token_id: int,
    append_answer: bool,
    context_length: int,
    batch_size: int,
    pad_strategy: str,
    subset: bool = False,
) -> Tuple[DataLoader, List[int]]:
    """Process BoolQ data and return a dataloader and labels"""
    data = load_dataset("google/boolq")[split]
    if subset:
        data = data.select(range(100))
    tensor_list = [boolq2tensor(x, tokenizer, append_answer) for x in data]
    tensor_data = tensorlist2padded(
        tensor_list, context_length, pad_token_id, pad_strategy
    )
    labels = torch.tensor([1 if x["answer"] else 0 for x in data], dtype=torch.long)
    dataset = TensorDataset(tensor_data, labels)
    return (
        DataLoader(dataset, batch_size=batch_size, shuffle=True),
        labels.cpu().tolist(),
    )


# ==========================
#     Training Functions
# ==========================


def model_inference(
    model: nn.Module,
    loader: DataLoader,
    indices: Union[List[int], None],
    device: torch.device,
    desc: str = "Validation",
) -> Tuple[List[int], List[torch.Tensor]]:
    """Run inference on a model and return predictions and logits"""
    model.eval()
    label_pred, logits = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            X, _ = batch
            X = X.to(device)
            outputs = model(X).logits[:, -1, :]
            if indices is not None:
                # Indices are used to select the logits for the correct tokens i.e [no, yes]
                outputs = outputs[:, indices]
            if outputs.shape[-1] == 1:
                # If the model is a binary classifier, use a sigmoid function to get the probability of the positive class
                preds = torch.where(F.sigmoid(outputs.reshape(-1)) > 0.5, 1, 0)
            else:
                # If the model is a multi-class classifier, use the argmax function to get the predicted class
                preds = outputs.argmax(dim=-1)
            label_pred.extend(preds.cpu().tolist())
            logits.extend(outputs.cpu().tolist())
    return label_pred, logits


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    save_model: bool = False,
    filename: str = "model.pt",
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Train a model and return training and validation losses and accuracies"""
    train_losses, dev_losses = [], []
    train_accuracies, dev_accuracies = [], []

    # Training loop
    progress_bar = tqdm(range(epochs), desc="Training", leave=True)
    for epoch in progress_bar:
        model.train()
        running_loss, running_correct, running_total = 0.0, 0, 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]", leave=False)
        for batch in train_bar:
            # Get batch
            X, y = batch
            X, y = X.to(device), y.to(device)

            # Forward pass
            scaler = torch.amp.GradScaler()
            with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
                logits = model(X).logits[:, -1, :]
                if isinstance(loss_fn, nn.BCEWithLogitsLoss):
                    loss = loss_fn(logits, y.float().reshape(-1, 1))
                else:
                    loss = loss_fn(logits, y.long())

            # Backward pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Update metrics
            running_loss += loss.item() * X.size(0)
            if logits.shape[-1] == 1:
                preds = torch.where(F.sigmoid(logits.reshape(-1)) > 0.5, 1, 0)
            else:
                preds = logits.argmax(dim=-1)
            running_correct += (preds == y).sum().item()
            running_total += X.size(0)
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
                X, y = batch
                X, y = X.to(device), y.to(device)

                # Forward pass
                with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
                    logits = model(X).logits[:, -1, :]
                    if isinstance(loss_fn, nn.BCEWithLogitsLoss):
                        loss = loss_fn(logits, y.float().reshape(-1, 1))
                    else:
                        loss = loss_fn(logits, y.long())

                # Update metrics
                if logits.shape[-1] == 1:
                    preds = torch.where(F.sigmoid(logits.reshape(-1)) > 0.5, 1, 0)
                else:
                    preds = logits.argmax(dim=-1)
                running_loss += loss.item() * X.size(0)
                running_correct += (preds == y).sum().item()
                running_total += X.size(0)
                val_bar.set_postfix(
                    loss=loss.item(), acc=running_correct / running_total
                )
        dev_losses.append(running_loss / running_total)
        dev_accuracies.append(running_correct / running_total)

    progress_bar.close()

    if save_model:
        print("Saving model...")
        torch.save(model.state_dict(), filename)

    return train_losses, dev_losses, train_accuracies, dev_accuracies


# ==========================
#     Helper Functions
# ==========================


def boolq2tensor(
    x: Dict[str, Any], tokenizer: GPT2Tokenizer, append_answer: bool = False
) -> torch.Tensor:
    """Convert BoolQ data to a tensor"""
    text = f"{x['passage']}.\n{x['question']}?\n"
    if append_answer:
        text += f"{'yes' if x['answer'] else 'no'}"
    return tokenizer.encode(text, return_tensors="pt")


def tensorlist2padded(
    tensorlist: List[torch.Tensor], length: int, pad_token_id: int, pad_strategy: str
) -> torch.Tensor:
    """Pad a list of tensors to a given length"""
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
    """Get a string representation of the metrics"""
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


# Function copied from a1_p2_murugan_116745378.py
def plot_loss_and_accuracy(
    train_losses: List[float],
    dev_losses: List[float],
    train_accuracies: List[float],
    dev_accuracies: List[float],
    filename: str,
) -> None:
    """Plot loss and accuracy for training and dev data"""
    # Create a figure with two subplots
    _, ax1 = plt.subplots(figsize=(10, 5))

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
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--save_model", action="store_true", default=False)
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--file_prefix", type=str, default="a3_p1_murugan_116745378")
    args = parser.parse_args()
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")
    dataloader_args = {
        "context_length": args.context_length,
        "batch_size": args.batch_size,
        "pad_strategy": "left",
        "subset": True,
    }
    optimizer_args = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
    }
    trainer_args = {
        "epochs": args.epochs,
        "save_model": args.save_model,
    }

    # Create and open output file
    print("Creating output file...")
    os.makedirs(args.save_dir, exist_ok=True)
    outfile = open(f"{args.save_dir}/{args.file_prefix}_OUTPUT.txt", "w")

    # Initialize tokenizer
    print("Initializing distilgpt2 model and tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    no_token_id, yes_token_id = tokenizer.encode("no")[0], tokenizer.encode("yes")[0]
    model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)

    # Load datasets
    print("Loading BoolQ validation dataset...")
    val_loader, val_labels = process_data_boolq(
        "validation", tokenizer, tokenizer.unk_token_id, False, **dataloader_args
    )

    # Zero-shot accuracy of distilgpt2 on BoolQ
    outfile.write("Checkpoint 1.1:\n")
    print("Evaluating Zero-shot accuracy of distilgpt2 on BoolQ...")
    label_pred, _ = model_inference(
        model, val_loader, [no_token_id, yes_token_id], device, "Zero-shot DistilGPT2"
    )
    outfile.write(get_metric_str(val_labels, label_pred) + "\n\n")

    # Finetune distilgpt2 on BoolQ
    outfile.write("Checkpoint 1.2:\n")
    print("Loading BoolQ training dataset...")
    train_loader, _ = process_data_boolq(
        "train", tokenizer, tokenizer.unk_token_id, True, **dataloader_args
    )

    print("Finetuning distilgpt2 on BoolQ...")
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_args)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.unk_token_id)
    losses = train_model(
        model,
        train_loader,
        val_loader,
        loss_fn,
        optimizer,
        device,
        filename=f"{args.save_dir}/{args.file_prefix}_model_distilgpt2.pt",
        **trainer_args,
    )
    plot_loss_and_accuracy(
        *losses, f"{args.save_dir}/{args.file_prefix}_loss_accuracy_distilgpt2.png"
    )
    outfile.write(
        f"Training plot saved to {args.save_dir}/{args.file_prefix}_loss_accuracy_distilgpt2.png\n\n"
    )

    # Zero-shot accuracy of finetuned distilgpt2 on BoolQ
    outfile.write("Checkpoint 1.3:\n")
    print("Evaluating Zero-shot accuracy of finetuned distilgpt2 on BoolQ...")
    label_pred, _ = model_inference(
        model,
        val_loader,
        [no_token_id, yes_token_id],
        device,
        "Instruction-tuned DistilGPT2",
    )
    outfile.write(get_metric_str(val_labels, label_pred) + "\n\n")

    outfile.write("Checkpoint 1.4:\n")
    print("Loading BoolQ training dataset...")
    tokenizer = RobertaTokenizer.from_pretrained("distilroberta-base")
    train_loader, _ = process_data_boolq(
        "train", tokenizer, tokenizer.pad_token_id, False, **dataloader_args
    )

    print("Loading DistilRoBERTa model...")
    model = AutoModelForCausalLM.from_pretrained("distilroberta-base", is_decoder=True)
    # TODO: autocast not compatible
    model.lm_head = nn.Sequential(
        nn.Linear(model.lm_head.dense.in_features, 1)  # , nn.Sigmoid()
    )
    model = model.to(device)
    # TODO: only optimize lm_head?
    optimizer = torch.optim.AdamW(model.lm_head.parameters(), **optimizer_args)
    # TODO: torch.nn.BCELoss is unsafe to autocast
    loss_fn = nn.BCEWithLogitsLoss()
    losses = train_model(
        model,
        train_loader,
        val_loader,
        loss_fn,
        optimizer,
        device,
        filename=f"{args.save_dir}/{args.file_prefix}_model_distilroberta.pt",
        **trainer_args,
    )
    plot_loss_and_accuracy(
        *losses, f"{args.save_dir}/{args.file_prefix}_loss_accuracy_distilroberta.png"
    )
    outfile.write(
        f"Training plot saved to {args.save_dir}/{args.file_prefix}_loss_accuracy_distilroberta.png\n\n"
    )
    label_pred, _ = model_inference(
        model,
        val_loader,
        None,
        device,
        "Finetuned DistilRoBERTa",
    )
    outfile.write(get_metric_str(val_labels, label_pred) + "\n\n")

    # Close output file
    print("Closing output file...")
    outfile.close()

    return None


if __name__ == "__main__":
    main()
