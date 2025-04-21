#! /usr/bin/env python3

import argparse
import os
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from einops import rearrange
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, GPT2LMHeadModel, RobertaModel

# Enable TF32 tensor cores for better performance
torch.set_float32_matmul_precision("high")

np.random.seed(0)
torch.manual_seed(0)

# ==========================
#     Data Functions
# ==========================


def get_distilgpt2_special_token_ids() -> Tuple[int, int, int]:
    """Get the token IDs for 'yes', 'no', and padding tokens in the distilgpt2 tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    no_token_id = tokenizer.encode("no")[0]
    yes_token_id = tokenizer.encode("yes")[0]
    pad_token_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    return no_token_id, yes_token_id, pad_token_id


# This function is also used in a1_p2_murugan_116745378.py
def load_and_preprocess_boolq(
    split: str, append_answer: bool = False
) -> Tuple[List[str], List[int]]:
    """Load and preprocess the BoolQ dataset for binary classification."""
    boolq_dataset = load_dataset("google/boolq")
    data = boolq_dataset[split]
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
def create_dataloader(
    data: List[str],
    labels: List[int],
    model_name: str,
    padding_side: str,
    truncation_side: str,
    context_length: int,
    batch_size: int,
) -> DataLoader:
    """Create a PyTorch DataLoader for the given dataset with specified tokenization parameters."""
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Note: We assign pad_token to unk_token if it is not assigned (for distilgpt2)
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
        # Note: We don't convert dtype below since it is handled in trainer class
        torch.tensor(labels),
    )
    dataloader = DataLoader(tensor_data, batch_size=batch_size, shuffle=False)
    return dataloader


# ==========================
#     Model Functions
# ==========================


# This function is also used in a1_p2_murugan_116745378.py
def load_distilgpt2_pretrained() -> nn.Module:
    """Load the pretrained distilgpt2 model."""
    model = GPT2LMHeadModel.from_pretrained("distilgpt2")
    return model


# This function is also used in a1_p2_murugan_116745378.py
def load_distilroberta_pretrained() -> nn.Module:
    """Load the pretrained distilroberta model with binary classification head."""
    model = RobertaModel.from_pretrained("distilroberta-base")

    # Note: We modify the pooler layer to be a dense layer from 768 to 1 with a sigmoid activation function
    # This allows us to use the model for binary classification as well as regression task with y range [0, 1]
    model.pooler.dense = nn.Linear(model.pooler.dense.in_features, 1)
    model.pooler.activation = nn.Sigmoid()
    return model


# ==========================
#     Training Functions
# ==========================


# This class is also used in a1_p2_murugan_116745378.py
class Trainer:
    """Reusable trainer class for all models and tasks."""

    def __init__(
        self,
        model: nn.Module,
        model_type: str,
        task_type: str,
        device: str,
        lr: float = 1e-5,
        weight_decay: float = 1e-3,
        loss_fn: Union[nn.Module, None] = None,
    ) -> None:
        self.model = model
        self.model_type = model_type  # "classification" or "regression"
        self.task_type = task_type  # "instruct-tuning" or "fine-tuning"
        # Initialize loss function based on model type and task type
        self.optimizer = AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # Note: We use different loss functions for different model types and task types
        # If loss_fn is provided, we use it instead of the default loss functions
        if loss_fn is not None:
            self.loss_fn = loss_fn
        elif self.model_type == "regression":
            self.loss_fn = nn.MSELoss()
        elif (
            self.model_type == "classification" and self.task_type == "instruct-tuning"
        ):
            self.loss_fn = nn.CrossEntropyLoss()
        elif self.model_type == "classification" and self.task_type == "fine-tuning":
            self.loss_fn = nn.BCELoss()

        self.device = device
        self.model.to(device)
        # Compile the model for better performance
        self.model = torch.compile(self.model, mode="reduce-overhead")

    def _forward(
        self, X: torch.Tensor, attn_mask: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.amp.autocast(device_type=self.device, dtype=torch.float16):
            if self.task_type == "instruct-tuning":
                # Remove the last token from the input and attention mask
                output = self.model(X[:, :-1], attention_mask=attn_mask[:, :-1])
                output = output.logits
                # Rearrange the output to be of shape (batch_size, sequence_length, num_classes)
                output = rearrange(output, "b s c -> b c s")
                y = X[:, 1:]
            else:  # task_type == "fine-tuning" for both classification and regression
                # Get the pooler layer output
                output = self.model(X, attention_mask=attn_mask)
                output = output.pooler_output

        loss = self._loss(output, y)
        return output, loss

    def _loss(self, output: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.task_type == "instruct-tuning" and self.model_type == "classification":
            loss = self.loss_fn(output.float(), y)
        else:  # task_type == "fine-tuning" for both classification and regression
            loss = self.loss_fn(output.float(), y.float().reshape(-1, 1))
        return loss

    # TODO: log stepwise loss and remove val loss tracking
    def train(
        self,
        train_dataloader: DataLoader,
        epochs: int,
        desc: str,
    ) -> Tuple[List[float], List[float]]:
        """Train the model for a given number of epochs."""
        epoch_losses, step_losses = [], []
        progress_bar = tqdm(range(epochs), desc=desc)
        for epoch in progress_bar:
            self.model.train()
            train_bar = tqdm(
                train_dataloader, desc=f"Epoch {epoch + 1} [Train]", leave=False
            )
            running_loss, running_total = 0, 0
            for batch in train_bar:
                X, attn_mask, y = batch
                X, attn_mask, y = (
                    X.to(self.device),
                    attn_mask.to(self.device),
                    y.to(self.device),
                )
                _, loss = self._forward(X, attn_mask, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * y.size(0)
                running_total += y.size(0)
                step_losses.append(loss.item())
                train_bar.set_postfix(loss=loss.item())
            epoch_losses.append(running_loss / running_total)

            progress_bar.set_postfix(train_loss=epoch_losses[-1])

        progress_bar.close()
        return epoch_losses, step_losses

    def inference(
        self, test_dataloader: DataLoader, indices: Union[List[int], None] = None
    ) -> List:
        """Inference the model on the test set."""
        self.model.eval()
        preds = []
        with torch.no_grad():
            test_bar = tqdm(test_dataloader, desc="Inference")
            for batch in test_bar:
                X, attn_mask, _ = batch
                X, attn_mask = (
                    X.to(self.device),
                    attn_mask.to(self.device),
                )
                output = self.model(X, attention_mask=attn_mask)
                if self.task_type == "instruct-tuning":
                    assert "logits" in output
                    # We only want the last token's logits during inference
                    output = output.logits[:, -1, :]
                else:  # task_type == "fine-tuning"
                    assert "pooler_output" in output
                    output = output.pooler_output
                if indices is not None:
                    # Note: Indices are used to select the logits for the correct tokens i.e [no, yes]
                    output = output[:, indices]
                preds.extend(output.cpu().tolist())
        preds = np.array(preds)

        # Note: We use different post-processing for different model types and task types
        if self.model_type == "regression":
            assert preds.shape[-1] == 1
            preds = preds[:, 0]
        else:
            if preds.shape[-1] == 1:
                # If the model is a binary classifier, use a decision threshold of 0.5
                preds = np.where(preds > 0.5, 1, 0).reshape(-1)
            else:
                # If the model is a multi-class classifier, use the argmax function to get the predicted class
                preds = np.argmax(preds, axis=-1)
        return preds.tolist()


# ==========================
#     Helper Functions
# ==========================


# This function is also used in a1_p2_murugan_116745378.py
def compute_binary_metrics(
    labels: List[int], predictions: List[int]
) -> Dict[str, float]:
    """Compute binary classification metrics for the given labels and predictions."""
    metrics = {
        "overall": {
            "acc": accuracy_score(labels, predictions),
            "f1": f1_score(labels, predictions, zero_division=0, average="macro"),
        },
        "positive": {
            "prec": precision_score(labels, predictions, pos_label=1, zero_division=0),
            "rec": recall_score(labels, predictions, pos_label=1, zero_division=0),
            "f1": f1_score(labels, predictions, pos_label=1, zero_division=0),
        },
        "negative": {
            "prec": precision_score(labels, predictions, pos_label=0, zero_division=0),
            "rec": recall_score(labels, predictions, pos_label=0, zero_division=0),
            "f1": f1_score(labels, predictions, pos_label=0, zero_division=0),
        },
    }
    return metrics


def format_metrics(metrics: Dict[str, Dict[str, float]]) -> str:
    """Format metrics dictionary into a readable string."""
    overall = metrics["overall"]
    positive = metrics["positive"]
    negative = metrics["negative"]

    return f"""Overall: acc: {overall['acc']:.3f}, f1: {overall['f1']:.3f}
    Yes: prec: {positive['prec']:.3f}, rec: {positive['rec']:.3f}, f1: {positive['f1']:.3f}
     No: prec: {negative['prec']:.3f}, rec: {negative['rec']:.3f}, f1: {negative['f1']:.3f}"""


# This function is also used in a1_p2_murugan_116745378.py
# BEGIN[ChatGPT][https://chatgpt.com/]"Python code + fix this code to accept epochlosses and steplosses (not same length) eg, say there are 500 steps and 5 epochs then i want the plot to show step loss and overlay epoch loss from 100(not zero), then 200 and so"
def plot_training_loss(
    epoch_losses: List[float],
    step_losses: List[float],
    title: str,
    filename: str,
) -> None:
    """Plot stepwise training loss with overlaid epoch loss markers."""

    _, ax = plt.subplots(figsize=(10, 6))
    steps = list(range(1, len(step_losses) + 1))

    # Plot stepwise loss
    ax.plot(steps, step_losses, color="blue", linestyle="-", label="Step Loss")

    # Overlay epoch loss every N steps
    steps_per_epoch = len(step_losses) // len(epoch_losses)
    epoch_steps = [steps_per_epoch * (i + 1) for i in range(len(epoch_losses))]

    ax.plot(
        epoch_steps,
        epoch_losses,
        color="red",
        linestyle="--",
        marker="o",
        label="Epoch Loss",
    )

    ax.set_xlabel("Steps")
    ax.set_ylabel("Loss")
    ax.grid(True)
    ax.legend()
    ax.set_title(title)
    ax.set_xticks(
        epoch_steps if len(step_losses) > 20 else steps
    )  # Avoid cluttering for large step counts

    print(f"Saving plot to {filename}...")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


# END[ChatGPT]


# ==========================
#     Main Function
# ==========================


def main() -> None:
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="script to run cse538 assignment 3 part 1"
    )
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--file_prefix", type=str, default="a3_p1_murugan_116745378")
    args = parser.parse_args()
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")
    base_dataloader_config = {
        "batch_size": args.batch_size,
    }
    gpt2_dataloader_config = {
        "model_name": "distilgpt2",
        "padding_side": "left",  # Left padding used for decoder models
        "truncation_side": "left",  # Left truncation used to preserve the question tokens
        "context_length": 200,  # Smaller context give better zero-shot & instruction-tuning results
        **base_dataloader_config,
    }
    roberta_dataloader_config = {
        "model_name": "distilroberta-base",
        "padding_side": "right",  # Right padding used for encoder models
        "truncation_side": "left",  # Left truncation used to preserve the question tokens
        "context_length": 256,
        **base_dataloader_config,
    }
    trainer_config = {
        "device": device,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
    }

    # Create and open output file
    print("Creating output file...")
    os.makedirs(args.save_dir, exist_ok=True)
    outfile = open(f"{args.save_dir}/{args.file_prefix}_OUTPUT.txt", "w")

    # Initialize model
    print("Initializing distilgpt2 model...")
    model = load_distilgpt2_pretrained()
    trainer = Trainer(model, "classification", "instruct-tuning", **trainer_config)

    # Load datasets
    print("Loading BoolQ validation dataset...")
    val_data, val_labels = load_and_preprocess_boolq("validation")
    val_loader = create_dataloader(val_data, val_labels, **gpt2_dataloader_config)

    # Get token ids for yes and no
    n_id, y_id, _ = get_distilgpt2_special_token_ids()

    # Zero-shot accuracy of distilgpt2 on BoolQ
    outfile.write("Checkpoint 1.1:\n")
    print("Evaluating Zero-shot accuracy of distilgpt2 on BoolQ...")
    label_pred = trainer.inference(val_loader, [n_id, y_id])
    results_dict = compute_binary_metrics(val_labels, label_pred)
    outfile.write(format_metrics(results_dict) + "\n\n")

    # Instruct-tune distilgpt2 on BoolQ
    outfile.write("Checkpoint 1.2:\n")
    print("Loading BoolQ training dataset...")
    train_data, train_labels = load_and_preprocess_boolq("train", append_answer=True)
    train_loader = create_dataloader(train_data, train_labels, **gpt2_dataloader_config)

    print("Instruct-tuning distilgpt2 on BoolQ...")
    losses = trainer.train(
        train_loader, args.epochs, "Instruct-tuning distilgpt2 on BoolQ"
    )
    title = "Instruct-tuning distilgpt2 on BoolQ"
    filename = f"{args.save_dir}/instructtuning_loss_distilgpt2.png"
    plot_training_loss(*losses, title, filename)
    outfile.write(f"Training plot saved to {filename}\n\n")

    # Zero-shot accuracy of instruct-tuned distilgpt2 on BoolQ
    outfile.write("Checkpoint 1.3:\n")
    print("Evaluating accuracy of instruct-tuned distilgpt2 on BoolQ...")
    label_pred = trainer.inference(val_loader, [n_id, y_id])
    results_dict = compute_binary_metrics(val_labels, label_pred)
    outfile.write(format_metrics(results_dict) + "\n\n")

    outfile.write("Checkpoint 1.4:\n")
    print("Loading BoolQ training dataset without answer...")
    train_data, train_labels = load_and_preprocess_boolq("train")
    train_loader = create_dataloader(
        train_data, train_labels, **roberta_dataloader_config
    )
    val_data, val_labels = load_and_preprocess_boolq("validation")
    val_loader = create_dataloader(val_data, val_labels, **roberta_dataloader_config)

    print("Loading DistilRoBERTa model...")
    model = load_distilroberta_pretrained()
    trainer = Trainer(model, "classification", "fine-tuning", **trainer_config)

    print("Finetuning distilroberta on BoolQ...")
    losses = trainer.train(
        train_loader, args.epochs, "Finetuning distilroberta on BoolQ"
    )
    title = "Finetuning distilroberta on BoolQ"
    filename = f"{args.save_dir}/finetuning_loss_distilroberta.png"
    plot_training_loss(*losses, title, filename)
    outfile.write(f"Training plot saved to {filename}\n\n")

    print("Evaluating accuracy of finetuned distilroberta on BoolQ...")
    label_pred = trainer.inference(val_loader)
    results_dict = compute_binary_metrics(val_labels, label_pred)
    outfile.write(format_metrics(results_dict) + "\n")

    # Close output file
    print("Closing output file...")
    outfile.close()

    return None


if __name__ == "__main__":
    main()
