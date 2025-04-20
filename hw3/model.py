import torch
from torch import nn
from transformers import GPT2LMHeadModel, RobertaModel


def get_distilgpt2_model() -> nn.Module:
    """Get the distilgpt2 model."""
    model = GPT2LMHeadModel.from_pretrained("distilgpt2")
    return model


def get_distilroberta_model() -> nn.Module:
    """Get the distilroberta model."""
    model = RobertaModel.from_pretrained("distilroberta-base")
    model.pooler.dense = nn.Linear(model.pooler.dense.in_features, 1)
    model.pooler.activation = nn.Sigmoid()
    return model


def distilroberta_rand_init(model: nn.Module) -> nn.Module:
    """Randomly initialize the distilroberta model."""

    def init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight)
            nn.init.normal_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            # nn.init.ones_(module.bias)
            nn.init.zeros_(module.bias)

    model.apply(init_weights)
    return model


def distilroberta_kqv_init(model: nn.Module) -> nn.Module:
    """Initialize the distilroberta model with K-Q-V initialization."""

    for name in ["encoder.layer.4", "encoder.layer.5"]:
        module = model.get_submodule(name)
        w = (module.attention.self.key.weight + module.attention.self.query.weight) / 2
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


def distilroberta_nores_init(model: nn.Module) -> nn.Module:
    """Initialize the distilroberta model with no residual connections."""
    for name in ["encoder.layer.4", "encoder.layer.5"]:
        module = model.get_submodule(name)
        nores_self_output = RobertaSelfOutputNoRes(model.config)
        nores_output = RobertaOutputNoRes(model.config)

        nores_self_output.load_state_dict(module.attention.output.state_dict())
        nores_output.load_state_dict(module.output.state_dict())

        module.attention.output = nores_self_output
        module.output = nores_output
    return model


if __name__ == "__main__":
    use_subset = False
    epochs = 5
    from collections import Counter

    from sklearn.metrics import accuracy_score, mean_absolute_error

    from data_functions import get_dataloader, process_boolq, process_sst
    from trainer import Trainer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("DEBUG: PROCESSING DATA")
    train_data, train_labels = process_boolq("train", subset=use_subset)
    train_loader = get_dataloader(
        train_data, train_labels, "distilroberta-base", "right", "left", 256, 16
    )
    val_data, val_labels = process_boolq("validation", subset=use_subset)
    val_loader = get_dataloader(
        val_data, val_labels, "distilroberta-base", "right", "left", 256, 16
    )

    print("DEBUG: INITIALIZING RANDOMLY INITIALIZED MODEL")
    model = get_distilroberta_model()
    model = distilroberta_rand_init(model)

    print("DEBUG: INFERENCING RANDOMLY INITIALIZED MODEL")
    trainer = Trainer(model, "classification", "fine-tuning", device)
    preds = trainer.inference(val_loader)
    print(f"DEBUG: COUNTER: {Counter(preds)}")
    print(f"DEBUG: ACCURACY: {accuracy_score(preds, val_labels)}")

    print("DEBUG: INITIALIZING BASELINE MODEL")
    model = get_distilroberta_model()

    print("DEBUG: TRAINING BASELINE MODEL")
    trainer = Trainer(model, "classification", "fine-tuning", device)
    trainer.train(train_loader, val_loader, epochs=epochs)

    print("DEBUG: INFERENCING BASELINE MODEL")
    preds = trainer.inference(val_loader)
    print(f"DEBUG: COUNTER: {Counter(preds)}")
    print(f"DEBUG: ACCURACY: {accuracy_score(preds, val_labels)}")

    print("DEBUG: INITIALIZING K-Q-V INITIALIZED MODEL")
    model = get_distilroberta_model()
    model = distilroberta_kqv_init(model)

    print("DEBUG: TRAINING K-Q-V INITIALIZED MODEL")
    trainer = Trainer(model, "classification", "fine-tuning", device)
    trainer.train(train_loader, val_loader, epochs=epochs)

    print("DEBUG: INFERENCING K-Q-V INITIALIZED MODEL")
    preds = trainer.inference(val_loader)
    print(f"DEBUG: COUNTER: {Counter(preds)}")
    print(f"DEBUG: ACCURACY: {accuracy_score(preds, val_labels)}")

    print("DEBUG: INITIALIZING NO RESIDUAL CONNECTIONS MODEL")
    model = get_distilroberta_model()
    model = distilroberta_nores_init(model)

    print("DEBUG: TRAINING NO RESIDUAL CONNECTIONS MODEL")
    trainer = Trainer(model, "classification", "fine-tuning", device)
    trainer.train(train_loader, val_loader, epochs=epochs)

    print("DEBUG: INFERENCING NO RESIDUAL CONNECTIONS MODEL")
    preds = trainer.inference(val_loader)
    print(f"DEBUG: COUNTER: {Counter(preds)}")
    print(f"DEBUG: ACCURACY: {accuracy_score(preds, val_labels)}")

    print("DEBUG: PROCESSING SST DATA")
    train_data, train_labels = process_sst("train", subset=use_subset)
    train_loader = get_dataloader(
        train_data, train_labels, "distilroberta-base", "right", "left", 128, 16
    )
    val_data, val_labels = process_sst("validation", subset=use_subset)
    val_loader = get_dataloader(
        val_data, val_labels, "distilroberta-base", "right", "left", 128, 16
    )
    test_data, test_labels = process_sst("test", subset=use_subset)
    test_loader = get_dataloader(
        test_data, test_labels, "distilroberta-base", "right", "left", 128, 16
    )

    print("DEBUG: INITIALIZING RANDOMLY INITIALIZED MODEL")
    model = get_distilroberta_model()
    model = distilroberta_rand_init(model)

    print("DEBUG: INFERENCING RANDOMLY INITIALIZED MODEL")
    trainer = Trainer(model, "regression", "fine-tuning", device)
    preds = trainer.inference(test_loader)
    print(f"DEBUG: MAE: {mean_absolute_error(preds, test_labels)}")

    print("DEBUG: INITIALIZING BASELINE MODEL")
    model = get_distilroberta_model()

    print("DEBUG: TRAINING BASELINE MODEL")
    trainer = Trainer(model, "regression", "fine-tuning", device)
    trainer.train(train_loader, val_loader, epochs=epochs)

    print("DEBUG: INFERENCING BASELINE MODEL")
    preds = trainer.inference(test_loader)
    print(f"DEBUG: MAE: {mean_absolute_error(preds, test_labels)}")

    print("DEBUG: INITIALIZING K-Q-V INITIALIZED MODEL")
    model = get_distilroberta_model()
    model = distilroberta_kqv_init(model)

    print("DEBUG: TRAINING K-Q-V INITIALIZED MODEL")
    trainer = Trainer(model, "regression", "fine-tuning", device)
    trainer.train(train_loader, val_loader, epochs=epochs)

    print("DEBUG: INFERENCING K-Q-V INITIALIZED MODEL")
    preds = trainer.inference(test_loader)
    print(f"DEBUG: MAE: {mean_absolute_error(preds, test_labels)}")

    print("DEBUG: INITIALIZING NO RESIDUAL CONNECTIONS MODEL")
    model = get_distilroberta_model()
    model = distilroberta_nores_init(model)

    print("DEBUG: TRAINING NO RESIDUAL CONNECTIONS MODEL")
    trainer = Trainer(model, "regression", "fine-tuning", device)
    trainer.train(train_loader, val_loader, epochs=epochs)

    print("DEBUG: INFERENCING NO RESIDUAL CONNECTIONS MODEL")
    preds = trainer.inference(test_loader)
    print(f"DEBUG: MAE: {mean_absolute_error(preds, test_labels)}")
