#! /usr/bin/env python3

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

from a3_p1_murugan_116745378 import (Trainer, create_dataloader,
                                     load_and_preprocess_boolq,
                                     load_distilroberta_pretrained,
                                     plot_training_loss)

# Enable TF32 tensor cores for better performance
torch.set_float32_matmul_precision("high")

np.random.seed(42)
torch.manual_seed(42)

# ==========================
#     Data Functions
# ==========================


def load_and_preprocess_sst(split: str) -> Tuple[List[str], List[float]]:
    """Load and preprocess the Stanford Sentiment Treebank dataset for sentiment analysis."""
    data = load_dataset("stanfordnlp/sst")[split]
    return data["sentence"], data["label"]


# ==========================
#   Model Initialization
# ==========================


def initialize_distilroberta_random(model: nn.Module) -> nn.Module:
    """Randomly initialize the distilroberta model weights."""

    def init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight)
            nn.init.normal_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.ones_(module.bias)

    model.apply(init_weights)
    return model


def initialize_distilroberta_shared_kqv(model: nn.Module) -> nn.Module:
    """Initialize the distilroberta model with shared K-Q-V weights."""
    # Modify 4th and 5th layers
    for name in ["encoder.layer.4", "encoder.layer.5"]:
        module = model.get_submodule(name)
        w = (module.attention.self.key.weight + module.attention.self.query.weight) / 2
        b = (module.attention.self.key.bias + module.attention.self.query.bias) / 2
        shared_linear_layer = nn.Linear(w.shape[1], w.shape[0])
        with torch.no_grad():
            shared_linear_layer.weight.copy_(w)
            shared_linear_layer.bias.copy_(b)

        # Replace the original key, query, and value layers with the shared linear layer
        module.attention.self.key = shared_linear_layer
        module.attention.self.query = shared_linear_layer
        module.attention.self.value = shared_linear_layer
    return model


# RobertaOutput modified to remove residual connection
class RobertaOutputNoRes(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)  # Removed residual connection
        return hidden_states


# RobertaSelfOutput modified to remove residual connection
class RobertaSelfOutputNoRes(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)  # Removed residual connection
        return hidden_states


def initialize_distilroberta_no_residual(model: nn.Module) -> nn.Module:
    """Initialize the distilroberta model with removed residual connections."""
    # Modify 4th and 5th layers
    for name in ["encoder.layer.4", "encoder.layer.5"]:
        module = model.get_submodule(name)

        # Create new self-output and output layers without residual connections
        nores_self_output = RobertaSelfOutputNoRes(model.config)
        nores_output = RobertaOutputNoRes(model.config)

        # Initialize the new layers with the same weights as the original layers
        nores_self_output.load_state_dict(module.attention.output.state_dict())
        nores_output.load_state_dict(module.output.state_dict())

        # Replace the original layers with the new ones
        module.attention.output = nores_self_output
        module.output = nores_output

    # Note: We create new classes for the self-output and output layers instead just
    # overwriting the forward method of the original layers to avoid modifying the
    # original class definitions
    return model


def initialize_distilroberta_enhanced_pooler(model: nn.Module) -> nn.Module:
    """Initialize the distilroberta model with an enhanced pooling layer."""

    # The original dense layer of the pooler is a single linear layer from the hidden size to 1
    # We replace it with a sequential layer of 2 linear layers with a tanh activation and a dropout
    model.pooler.dense = nn.Sequential(
        nn.Linear(model.pooler.dense.in_features, model.pooler.dense.in_features // 2),
        nn.Tanh(),
        nn.Dropout(0.1),
        nn.Linear(model.pooler.dense.in_features // 2, 1),
    )
    return model


# ==========================
#     Helper Functions
# ==========================


def compute_binary_metrics(
    labels: List[int], predictions: List[int]
) -> Dict[str, float]:
    """Compute binary classification metrics for the given labels and predictions."""
    metrics = {
        "acc": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, zero_division=0, average="macro"),
    }
    return metrics


def compute_regression_metrics(
    labels: List[float], predictions: List[float]
) -> Dict[str, float]:
    """Compute regression metrics for the given labels and predictions."""
    metrics = {
        "mae": mean_absolute_error(labels, predictions),
        "r": pearsonr(labels, predictions)[0],
    }
    return metrics


def format_binary_metrics(metric_dict: dict) -> str:
    return f"""boolq validation set:
 distilRB-rand: overall acc: {metric_dict["rand"]["acc"]:.3f}, f1: {metric_dict["rand"]["f1"]:.3f}
 distilroberta: overall acc: {metric_dict["base"]["acc"]:.3f}, f1: {metric_dict["base"]["f1"]:.3f}
  distilRB-KQV: overall acc: {metric_dict["kqv"]["acc"]:.3f}, f1: {metric_dict["kqv"]["f1"]:.3f}
distilRB-nores: overall acc: {metric_dict["nores"]["acc"]:.3f}, f1: {metric_dict["nores"]["f1"]:.3f}"""


def format_regression_metrics(metric_dict: dict) -> str:
    return f"""Validation: mae: {metric_dict["base"]["mae_val"]:.3f}, r: {metric_dict["base"]["r_val"]:.3f}
      Test: mae: {metric_dict["base"]["mae"]:.3f}, r: {metric_dict["base"]["r"]:.3f}

SST test set:
 distilRB-rand: mae: {metric_dict["rand"]["mae"]:.3f}, r: {metric_dict["rand"]["r"]:.3f}
  distilRB-KQV: mae: {metric_dict["kqv"]["mae"]:.3f}, r: {metric_dict["kqv"]["r"]:.3f}
distilRB-nores: mae: {metric_dict["nores"]["mae"]:.3f}, r: {metric_dict["nores"]["r"]:.3f}"""


def format_improved_metrics(
    binary_metric_dict: dict, regression_metric_dict: dict
) -> str:
    # Note: We print distilroberta metrics for reference as it is the best performing givenmodel among the ones we fine-tuned
    return f"""boolq validation set:
distilroberta: overall acc: {binary_metric_dict["base"]["acc"]:.3f}, f1: {binary_metric_dict["base"]["f1"]:.3f}
distilRB-improved: overall acc: {binary_metric_dict["improved"]["acc"]:.3f}, f1: {binary_metric_dict["improved"]["f1"]:.3f}

SST test set:
distilroberta: mae: {regression_metric_dict["base"]["mae"]:.3f}, r: {regression_metric_dict["base"]["r"]:.3f}
distilRB-improved: mae: {regression_metric_dict["improved"]["mae"]:.3f}, r: {regression_metric_dict["improved"]["r"]:.3f}"""


# ==========================
#     Main Function
# ==========================


def main() -> None:
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="script to run cse538 assignment 3 part 2"
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--boolq_batch_size", type=int, default=24)
    parser.add_argument("--sst_batch_size", type=int, default=32)
    parser.add_argument("--boolq_context_length", type=int, default=256)
    parser.add_argument("--sst_context_length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--file_prefix", type=str, default="a3_p2_murugan_116745378")
    args = parser.parse_args()
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")
    roberta_dataloader_config = {
        "model_name": "distilroberta-base",
        "padding_side": "right",  # Right padding used for encoder models
        "truncation_side": "left",  # Left truncation used for preserve the question tokens
    }
    boolq_config = {
        "context_length": args.boolq_context_length,
        "batch_size": args.boolq_batch_size,
    }
    sst_config = {
        "context_length": args.sst_context_length,
        "batch_size": args.sst_batch_size,
    }
    trainer_args = {
        "device": device,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
    }

    # Create and open output file
    print("Creating output file...")
    os.makedirs(args.save_dir, exist_ok=True)
    outfile = open(f"{args.save_dir}/{args.file_prefix}_OUTPUT.txt", "w")

    print("Creating model function map...")
    model_func_map = {
        "rand": initialize_distilroberta_random,
        "kqv": initialize_distilroberta_shared_kqv,
        "nores": initialize_distilroberta_no_residual,
        "improved": initialize_distilroberta_enhanced_pooler,
    }

    # Load data
    outfile.write("Checkpoint 2.2:\n")
    print("Loading BoolQ train and validation data...")
    train_data, train_labels = load_and_preprocess_boolq("train")
    train_loader = create_dataloader(
        train_data,
        train_labels,
        **boolq_config,
        **roberta_dataloader_config,
    )
    val_data, val_labels = load_and_preprocess_boolq("validation")
    val_loader = create_dataloader(
        val_data,
        val_labels,
        **boolq_config,
        **roberta_dataloader_config,
    )

    print("Fine-tuning models on BoolQ...")
    binary_metric_dict = {}
    for model_name in ["rand", "base", "kqv", "nores"]:
        print(f"Fine-tuning distilroberta-{model_name} on BoolQ...")
        model = load_distilroberta_pretrained()
        if model_name != "base":
            model = model_func_map[model_name](model)
        trainer = Trainer(model, "classification", "fine-tuning", **trainer_args)

        # Define loss function and optimizer
        if model_name != "rand":
            losses = trainer.train(
                train_loader, args.epochs, f"Fine-tuning {model_name} on boolq"
            )

        print(f"Evaluating distilroberta-{model_name} on BoolQ validation set...")
        label_pred = trainer.inference(val_loader)
        binary_metric_dict[model_name] = compute_binary_metrics(val_labels, label_pred)
    outfile.write(format_binary_metrics(binary_metric_dict) + "\n\n")

    outfile.write("Checkpoint 2.3:\n")
    print("Loading SST train, validation, and test data...")
    train_data, train_values = load_and_preprocess_sst("train")
    train_loader = create_dataloader(
        train_data,
        train_values,
        **sst_config,
        **roberta_dataloader_config,
    )
    val_data, val_values = load_and_preprocess_sst("validation")
    val_loader = create_dataloader(
        val_data,
        val_values,
        **sst_config,
        **roberta_dataloader_config,
    )
    test_data, test_values = load_and_preprocess_sst("test")
    test_loader = create_dataloader(
        test_data,
        test_values,
        **sst_config,
        **roberta_dataloader_config,
    )

    # Fine-tuning models on SST
    regression_metric_dict = {}
    print("Fine-tuning models on SST...")
    for model_name in ["rand", "base", "kqv", "nores"]:
        print(f"Fine-tuning distilroberta-{model_name} on SST...")
        model = load_distilroberta_pretrained()
        if model_name != "base":
            model = model_func_map[model_name](model)
        trainer = Trainer(model, "regression", "fine-tuning", **trainer_args)

        # Only fine-tune the model if it is not the random initialized model
        if model_name != "rand":
            losses = trainer.train(
                train_loader, args.epochs, f"Fine-tuning {model_name} on sst"
            )
            if model_name == "base":
                # Plot the training loss for the base model
                title = "Finetuning distilroberta-base on SST"
                filename = f"{args.save_dir}/finetuning_loss_plot_distilroberta_base_regression.png"
                plot_training_loss(*losses, title, filename)
                outfile.write(f"Training plot saved to {filename}\n")

        print(f"Evaluating distilroberta-{model_name} on SST test set...")
        output = trainer.inference(test_loader)
        regression_metric_dict[model_name] = compute_regression_metrics(
            test_values, output
        )
        if model_name == "base":
            # Evaluate the base model on the validation set
            print(f"Evaluating distilroberta-{model_name} on SST validation set...")
            output = trainer.inference(val_loader)
            val_metric_dict = compute_regression_metrics(val_values, output)
            regression_metric_dict[model_name]["mae_val"] = val_metric_dict["mae"]
            regression_metric_dict[model_name]["r_val"] = val_metric_dict["r"]

    outfile.write(format_regression_metrics(regression_metric_dict) + "\n\n")

    # Fine-tuning improved model on BoolQ
    outfile.write("Checkpoint 2.4: Extra Credit\n")
    print("Loading BoolQ train and validation data for improved model...")
    train_data, train_values = load_and_preprocess_boolq("train")
    train_loader = create_dataloader(
        train_data,
        train_values,
        **boolq_config,
        **roberta_dataloader_config,
    )
    val_data, val_values = load_and_preprocess_boolq("validation")
    val_loader = create_dataloader(
        val_data,
        val_values,
        **boolq_config,
        **roberta_dataloader_config,
    )

    print("Fine-tuning improved model on BoolQ...")
    model = load_distilroberta_pretrained()
    model = initialize_distilroberta_enhanced_pooler(model)
    trainer = Trainer(model, "classification", "fine-tuning", **trainer_args)
    losses = trainer.train(train_loader, args.epochs, f"Fine-tuning improved on boolq")

    print(f"Evaluating improved model on BoolQ validation set...")
    label_pred = trainer.inference(val_loader)
    binary_metric_dict["improved"] = compute_binary_metrics(val_values, label_pred)

    # Fine-tuning improved model on SST
    print("Loading SST train and test data for improved model...")
    train_data, train_values = load_and_preprocess_sst("train")
    train_loader = create_dataloader(
        train_data,
        train_values,
        **sst_config,
        **roberta_dataloader_config,
    )
    test_data, test_values = load_and_preprocess_sst("test")
    test_loader = create_dataloader(
        test_data,
        test_values,
        **sst_config,
        **roberta_dataloader_config,
    )

    print("Fine-tuning improved model on SST...")
    model = load_distilroberta_pretrained()
    model = initialize_distilroberta_enhanced_pooler(model)
    trainer = Trainer(model, "regression", "fine-tuning", **trainer_args)
    losses = trainer.train(train_loader, args.epochs, f"Fine-tuning improved on sst")

    print(f"Evaluating improved model on SST test set...")
    output = trainer.inference(test_loader)
    regression_metric_dict["improved"] = compute_regression_metrics(test_values, output)
    outfile.write(format_improved_metrics(binary_metric_dict, regression_metric_dict))

    # Close output file
    print("Closing output file...")
    outfile.close()

    return None


if __name__ == "__main__":
    main()
