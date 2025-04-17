#! /usr/bin/env python3

import argparse
import os

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, RobertaModel

from a3_p1_murugan_116745378 import (model_inference, plot_loss_and_accuracy,
                                     process_data_boolq, train_model)

torch.manual_seed(0)

# ==========================
#   Model Initialization
# ==========================


def get_distilroberta() -> nn.Module:
    # TODO: Check if model with LM head should be used for this task
    model = RobertaModel.from_pretrained("distilroberta-base")
    return model


def get_distilroberta_rand() -> nn.Module:
    model = RobertaModel.from_pretrained("distilroberta-base")
    for name, module in model.named_modules():
        # TODO: check if this is correct or this is to be done for all layers
        if "roberta.encoder.layer.4" in name or "roberta.encoder.layer.5" in name:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight)
                nn.init.normal_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    return model


def get_distilroberta_kqv() -> nn.Module:
    model = RobertaModel.from_pretrained("distilroberta-base")
    for name, module in model.named_modules():
        if name in {"roberta.encoder.layer.4", "roberta.encoder.layer.5"}:
            w = (
                module.attention.self.key.weight + module.attention.self.query.weight
            ) / 2
            b = (module.attention.self.key.bias + module.attention.self.query.bias) / 2
            shared_linear_layer = nn.Linear(w.shape[1], w.shape[0])
            with torch.no_grad():
                shared_linear_layer.weight.copy_(w)
                shared_linear_layer.bias.copy_(b)
            module.attention.self.key = shared_linear_layer
            module.attention.self.query = shared_linear_layer
            module.attention.self.value = shared_linear_layer
    return model


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


def get_distilroberta_nores() -> nn.Module:
    model = RobertaModel.from_pretrained("distilroberta-base", is_decoder=True)

    for name, module in model.named_modules():
        if name in {"roberta.encoder.layer.4", "roberta.encoder.layer.5"}:
            nores_self_output = RobertaSelfOutputNoRes(model.config)
            nores_output = RobertaOutputNoRes(model.config)

            nores_self_output.load_state_dict(module.attention.output.state_dict())
            nores_output.load_state_dict(module.output.state_dict())

            module.attention.output = nores_self_output
            module.output = nores_output
    return model


# ==========================
#     Helper Functions
# ==========================


def get_boolq_validation_metric_str(metric_dict: dict) -> str:
    return f"""boolq validation set:
 distilRB-rand: overall acc: {metric_dict["rand"]["accuracy"]:.3f}, f1: {metric_dict["rand"]["f1"]:.3f}
 distilroberta: overall acc: {metric_dict["base"]["accuracy"]:.3f}, f1: {metric_dict["base"]["f1"]:.3f}
  distilRB-KQV: overall acc: {metric_dict["kqv"]["accuracy"]:.3f}, f1: {metric_dict["kqv"]["f1"]:.3f}
distilRB-nores: overall acc: {metric_dict["nores"]["accuracy"]:.3f}, f1: {metric_dict["nores"]["f1"]:.3f}
"""


# ==========================
#     Main Function
# ==========================


def main() -> None:
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="script to run cse538 assignment 3 part 2"
    )
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--save_model", action="store_true", default=False)
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--file_prefix", type=str, default="a3_p2_murugan_116745378")
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
        "pad_strategy": "right",
        "subset": True,
    }
    optimizer_args = {"lr": args.lr, "weight_decay": args.weight_decay}
    trainer_args = {
        "epochs": args.epochs,
        "save_model": args.save_model,
    }

    # Create and open output file
    print("Creating output file...")
    os.makedirs(args.save_dir, exist_ok=True)
    outfile = open(f"{args.save_dir}/{args.file_prefix}_OUTPUT.txt", "w")

    # Load data
    outfile.write("Checkpoint 2.2:\n")
    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
    train_loader, _ = process_data_boolq(
        "train", tokenizer, tokenizer.pad_token_id, False, **dataloader_args
    )
    val_loader, val_labels = process_data_boolq(
        "validation", tokenizer, tokenizer.pad_token_id, False, **dataloader_args
    )

    model_func_map = {
        "rand": get_distilroberta_rand,
        "base": get_distilroberta,
        "kqv": get_distilroberta_kqv,
        "nores": get_distilroberta_nores,
    }

    metric_dict = {}
    for model_name in ["rand", "base", "kqv", "nores"]:
        model = model_func_map[model_name]()
        # TODO: autocast seems to be incompatible with BCELoss
        model.pooler.dense = nn.Linear(model.config.hidden_size, 1)
        model.pooler.activation = nn.Identity()
        model.to(device)

        # Define loss function and optimizer
        if model_name != "rand":
            loss_fn = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.AdamW(model.pooler.parameters(), **optimizer_args)

            losses = train_model(
                model,
                train_loader,
                val_loader,
                loss_fn,
                optimizer,
                device,
                **trainer_args,
            )
            plot_loss_and_accuracy(
                *losses,
                f"{args.save_dir}/{args.file_prefix}_loss_accuracy_distilroberta_{model_name}.png",
            )
        label_pred, _ = model_inference(
            model, val_loader, None, device, f"Finetuned DistilRoBERTa {model_name}"
        )
        metric_dict[model_name] = {
            "accuracy": accuracy_score(val_labels, label_pred),
            "f1": f1_score(val_labels, label_pred, average="macro"),
        }
    outfile.write(get_boolq_validation_metric_str(metric_dict) + "\n\n")

    # Close output file
    print("Closing output file...")
    outfile.close()

    return None


if __name__ == "__main__":
    main()

    input = torch.randint(0, 50257, (5, 256))

    model = get_distilroberta_rand()
    output = model(input).logits
    print(f"input shape: {input.shape}")
    print(f"output shape: {output.shape}")

    model = get_distilroberta_kqv()
    output = model(input).logits
    print(f"input shape: {input.shape}")
    print(f"output shape: {output.shape}")

    model = get_distilroberta_nores()
    output = model(input).logits
    print(f"input shape: {input.shape}")
    print(f"output shape: {output.shape}")
