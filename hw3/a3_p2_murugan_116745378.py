#! /usr/bin/env python3

import argparse

import torch
import torch.nn as nn
from transformers import AutoTokenizer, RobertaModel

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
#     Main Function
# ==========================


def main() -> None:
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="script to run cse538 assignment 3 part 2"
    )
    args = parser.parse_args()

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
