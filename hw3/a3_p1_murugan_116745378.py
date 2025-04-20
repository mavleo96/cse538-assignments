#! /usr/bin/env python3

import argparse
import os
from typing import Any, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, GPT2LMHeadModel, RobertaModel

torch.manual_seed(0)

# ==========================
#     Data Functions
# ==========================


# This function is also used in a1_p2_murugan_116745378.py
def process_boolq(
    split: str, append_answer: bool = False, subset: bool = False
) -> Tuple[List[str], List[int]]:
    """Process the BoolQ dataset"""
    boolq_dataset = load_dataset("google/boolq")
    data = boolq_dataset[split]
    if subset:
        data = data.select(range(100))
    if append_answer:
        li = [
            f"{x['passage']}.\n{x['question']}?\n{'yes' if x['answer'] else 'no'}"
            for x in data
        ]
    else:
        li = [f"{x['passage']}.\n{x['question']}?\n" for x in data]
    labels = [1 if x["answer"] else 0 for x in data]
    return li, labels


# This function is also used in a1_p2_murugan_116745378.py
def get_dataloader(
    data: List[str],
    labels: List[int],
    model: str,
    padding_side: str,
    truncation_side: str,
    context_length: int,
    batch_size: int,
) -> DataLoader:
    """Get dataloader for a dataset"""
    # Initialise tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = padding_side
    tokenizer.truncation_side = truncation_side

    # Tokenize data
    tokenized_data = tokenizer(
        data,
        return_tensors="pt",
        padding=True,
        max_length=context_length,
        truncation=True,
    )

    # Create tensor dataset and dataloader
    tensor_data = TensorDataset(
        tokenized_data["input_ids"],
        tokenized_data["attention_mask"],
        torch.tensor(labels, dtype=torch.long),
    )
    dataloader = DataLoader(tensor_data, batch_size=batch_size, shuffle=True)
    return dataloader


def get_yes_no_pad_token_ids() -> Tuple[int, int, int]:
    """Get the token ids for yes, no, and pad token for distilgpt2"""
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    no_token_id = tokenizer.encode("no")[0]
    yes_token_id = tokenizer.encode("yes")[0]
    pad_token_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.unk_token_id
    )
    return no_token_id, yes_token_id, pad_token_id


# ==========================
#     Training Functions
# ==========================


def model_inference(
    model: nn.Module,
    loader: DataLoader,
    indices: Union[List[int], None],
    device: torch.device,
    desc: str = "Validation",
    regression: bool = False,
) -> List[int]:
    """Run inference on a model"""
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            X, attn_mask, _ = batch
            X, attn_mask = X.to(device), attn_mask.to(device)
            outputs = model(X, attention_mask=attn_mask)
            if "logits" in outputs:
                outputs = outputs.logits[:, -1, :]
            elif "pooler_output" in outputs:
                outputs = outputs.pooler_output
            else:
                raise ValueError(f"Unknown output type: {outputs.keys()}")
            if indices is not None:
                # Indices are used to select the logits for the correct tokens i.e [no, yes]
                outputs = outputs[:, indices]
            preds.extend(outputs.cpu().tolist())

    preds = np.array(preds)
    if regression:
        assert preds.shape[-1] == 1
        preds = preds[:, 0]
    else:
        if preds.shape[-1] == 1:
            # If the model is a binary classifier, use a sigmoid function to get the probability of the positive class
            preds = 1 / (1 + np.exp(-preds[:, 0]))
            preds = np.where(preds > 0.5, 1, 0)
        else:
            # If the model is a multi-class classifier, use the argmax function to get the predicted class
            preds = np.argmax(preds, axis=-1)
    return preds.tolist()


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
        running_loss, running_total = 0.0, 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]", leave=False)
        for batch in train_bar:
            # Get batch
            X, attn_mask, y = batch
            X, attn_mask, y = X.to(device), attn_mask.to(device), y.to(device)

            # Forward pass
            scaler = torch.amp.GradScaler()
            with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
                outputs = model(X, attention_mask=attn_mask)
                if "logits" in outputs:
                    outputs = outputs.logits[:, -1, :]
                elif "pooler_output" in outputs:
                    outputs = outputs.pooler_output
                else:
                    raise ValueError(f"Unknown output type: {outputs.keys()}")
                if isinstance(loss_fn, (nn.BCEWithLogitsLoss, nn.MSELoss)):
                    loss = loss_fn(outputs, y.float().reshape(-1, 1))
                elif isinstance(loss_fn, nn.CrossEntropyLoss):
                    loss = loss_fn(outputs, y.long())
                else:
                    raise ValueError(f"Unknown loss function: {loss_fn}")

            # Backward pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # TODO: add accuracy / mae score
            # Update metrics
            running_loss += loss.item() * X.size(0)
            running_total += X.size(0)
            train_bar.set_postfix(loss=loss.item())

        train_losses.append(running_loss / running_total)

        # Validation loop
        model.eval()
        running_loss, running_total = 0.0, 0
        with torch.no_grad():
            val_bar = tqdm(
                val_loader, desc=f"Epoch {epoch + 1} [Validation]", leave=False
            )
            for batch in val_bar:
                # Get batch
                X, attn_mask, y = batch
                X, attn_mask, y = X.to(device), attn_mask.to(device), y.to(device)

                # Forward pass
                with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
                    outputs = model(X, attention_mask=attn_mask)
                    if "logits" in outputs:
                        outputs = outputs.logits[:, -1, :]
                    elif "pooler_output" in outputs:
                        outputs = outputs.pooler_output
                    else:
                        raise ValueError(f"Unknown output type: {outputs.keys()}")
                    if isinstance(loss_fn, (nn.BCEWithLogitsLoss, nn.MSELoss)):
                        loss = loss_fn(outputs, y.float().reshape(-1, 1))
                    elif isinstance(loss_fn, nn.CrossEntropyLoss):
                        loss = loss_fn(outputs, y.long())
                    else:
                        raise ValueError(f"Unknown loss function: {loss_fn}")

                # Update metrics
                running_loss += loss.item() * X.size(0)
                running_total += X.size(0)
                val_bar.set_postfix(loss=loss.item())
        dev_losses.append(running_loss / running_total)

    progress_bar.close()

    if save_model:
        print("Saving model...")
        torch.save(model.state_dict(), filename)

    return train_losses, dev_losses, train_accuracies, dev_accuracies


# ==========================
#     Helper Functions
# ==========================


# TODO: change this to accept a dictionary of metrics
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
    parser.add_argument("--use_subset", action="store_true", default=False)
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
    }
    distilgpt2_dataloader_args = {
        "model": "distilgpt2",
        "padding_side": "left",
        "truncation_side": "left",
        **dataloader_args,
    }
    distilroberta_dataloader_args = {
        "model": "distilroberta-base",
        "padding_side": "right",
        "truncation_side": "left",
        **dataloader_args,
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

    # Initialize model
    print("Initializing distilgpt2 model...")
    model = GPT2LMHeadModel.from_pretrained("distilgpt2").to(device)

    # Load datasets
    print("Loading BoolQ validation dataset...")
    val_data, val_labels = process_boolq(
        "validation", append_answer=False, subset=args.use_subset
    )
    val_loader = get_dataloader(val_data, val_labels, **distilgpt2_dataloader_args)

    # Get token ids for yes and no
    no_token_id, yes_token_id, pad_token_id = get_yes_no_pad_token_ids()

    # Zero-shot accuracy of distilgpt2 on BoolQ
    outfile.write("Checkpoint 1.1:\n")
    print("Evaluating Zero-shot accuracy of distilgpt2 on BoolQ...")
    label_pred = model_inference(
        model, val_loader, [no_token_id, yes_token_id], device, "Zero-shot DistilGPT2"
    )
    outfile.write(get_metric_str(val_labels, label_pred) + "\n\n")

    # Finetune distilgpt2 on BoolQ
    outfile.write("Checkpoint 1.2:\n")
    print("Loading BoolQ training dataset...")
    train_data, train_labels = process_boolq(
        "train", append_answer=True, subset=args.use_subset
    )
    train_loader = get_dataloader(
        train_data, train_labels, **distilgpt2_dataloader_args
    )

    print("Finetuning distilgpt2 on BoolQ...")
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_args)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    # TODO: model overfits to training data, need to train over full sequence
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
    label_pred = model_inference(
        model,
        val_loader,
        [no_token_id, yes_token_id],
        device,
        "Instruction-tuned DistilGPT2",
    )
    outfile.write(get_metric_str(val_labels, label_pred) + "\n\n")

    outfile.write("Checkpoint 1.4:\n")
    print("Loading BoolQ training dataset...")
    train_data, train_labels = process_boolq(
        "train", append_answer=False, subset=args.use_subset
    )
    train_loader = get_dataloader(
        train_data, train_labels, **distilroberta_dataloader_args
    )
    val_data, val_labels = process_boolq(
        "validation", append_answer=False, subset=args.use_subset
    )
    val_loader = get_dataloader(val_data, val_labels, **distilroberta_dataloader_args)

    print("Loading DistilRoBERTa model...")
    # TODO: Check if LMHead should be used instead of pooler
    model = RobertaModel.from_pretrained("distilroberta-base")
    model.pooler.dense = nn.Linear(model.pooler.dense.in_features, 1)
    # TODO: autocast not compatible with BCE loss
    model.pooler.activation = nn.Identity()
    model = model.to(device)
    # TODO: only optimize pooler?
    optimizer = torch.optim.AdamW(model.pooler.parameters(), **optimizer_args)
    # TODO: torch.nn.BCELoss is unsafe to autocast
    loss_fn = nn.BCEWithLogitsLoss()
    # TODO: model seems to be predicting either 0 or 1 for all examples
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
    label_pred = model_inference(
        model,
        val_loader,
        None,
        device,
        "Finetuned DistilRoBERTa",
    )
    outfile.write(get_metric_str(val_labels, label_pred) + "\n")

    # Close output file
    print("Closing output file...")
    outfile.close()

    return None


if __name__ == "__main__":
    main()
