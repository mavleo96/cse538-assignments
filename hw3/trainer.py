from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import tqdm
from einops import rearrange
from sklearn.metrics import accuracy_score, mean_absolute_error
from torch.optim import AdamW
from torch.utils.data import DataLoader


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
        save_model: bool = False,
    ) -> None:
        self.model = model
        self.model_type = model_type  # "classification" or "regression"
        self.task_type = task_type  # "instruction-tuning" or "fine-tuning"
        # Initialize loss function based on model type and task type
        self.optimizer = AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        if self.model_type == "regression":
            self.loss_fn = nn.MSELoss()
        elif (
            self.model_type == "classification"
            and self.task_type == "instruction-tuning"
        ):
            self.loss_fn = nn.CrossEntropyLoss()
        elif self.model_type == "classification" and self.task_type == "fine-tuning":
            self.loss_fn = nn.BCELoss()

        self.device = device
        self.model.to(device)

    def _forward(
        self, X: torch.Tensor, attn_mask: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.amp.autocast(device_type=self.device, dtype=torch.bfloat16):
            if self.task_type == "instruction-tuning":
                # Remove the last token from the input and attention mask
                output = self.model(X[:, :-1], attention_mask=attn_mask[:, :-1])
                output = output.logits
                # Rearrange the output to be of shape (batch_size, sequence_length, num_classes)
                output = rearrange(output, "b s c -> b c s")
                y = X[:, 1:]
            else:  # task_type == "fine-tuning"
                # Get the pooler layer output
                output = self.model(X, attention_mask=attn_mask)
                output = output.pooler_output

        loss = self._loss(output, y)
        return output, loss

    def _loss(self, output: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if (
            self.task_type == "instruction-tuning"
            and self.model_type == "classification"
        ):
            loss = self.loss_fn(output.float(), y)
        else:  # task_type == "fine-tuning"
            loss = self.loss_fn(output.float(), y.float().reshape(-1, 1))
        return loss

    def train(
        self, train_dataloader: DataLoader, val_dataloader: DataLoader, epochs: int
    ) -> Tuple[List[float], List[float]]:
        """Train the model for a given number of epochs."""
        train_losses, val_losses = [], []
        progress_bar = tqdm.tqdm(range(epochs))
        for epoch in progress_bar:
            self.model.train()
            train_bar = tqdm.tqdm(
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
                output, loss = self._forward(X, attn_mask, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * y.size(0)
                running_total += y.size(0)
                train_bar.set_postfix(loss=loss.item())
            train_losses.append(running_loss / running_total)

            # Validation Loop
            if val_dataloader is not None:
                val_loss = self._validation(val_dataloader, epoch)
                val_losses.append(val_loss)

            progress_bar.set_postfix(
                train_loss=train_losses[-1],
                val_loss=val_losses[-1] if val_dataloader is not None else None,
            )

        progress_bar.close()
        return train_losses, val_losses

    def _validation(self, val_dataloader: DataLoader, epoch: int) -> float:
        """Validate the model on the validation set"""
        self.model.eval()
        with torch.no_grad():
            val_bar = tqdm.tqdm(
                val_dataloader, desc=f"Epoch {epoch + 1} [Validation]", leave=False
            )
            running_loss, running_total = 0, 0
            for batch in val_bar:
                X, attn_mask, y = batch
                X, attn_mask, y = (
                    X.to(self.device),
                    attn_mask.to(self.device),
                    y.to(self.device),
                )
                output, loss = self._forward(X, attn_mask, y)
                val_bar.set_postfix(loss=loss.item())
                running_loss += loss.item() * y.size(0)
                running_total += y.size(0)

        return running_loss / running_total

    def inference(
        self, test_dataloader: DataLoader, indices: Union[List[int], None] = None
    ) -> List:
        self.model.eval()
        preds = []
        with torch.no_grad():
            for batch in test_dataloader:
                X, attn_mask, _ = batch
                X, attn_mask = (
                    X.to(self.device),
                    attn_mask.to(self.device),
                )
                output = self.model(X, attention_mask=attn_mask)
                if self.task_type == "instruction-tuning":
                    assert "logits" in output
                    # We only want the last token's logits during inference
                    output = output.logits[:, -1, :]
                else:  # task_type == "fine-tuning"
                    assert "pooler_output" in output
                    output = output.pooler_output
                if indices is not None:
                    # Indices are used to select the logits for the correct tokens i.e [no, yes]
                    output = output[:, indices]
                preds.extend(output.cpu().tolist())
        preds = np.array(preds)
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


if __name__ == "__main__":
    use_subset = True
    from collections import Counter

    from sklearn.metrics import accuracy_score, mean_absolute_error
    from transformers import GPT2LMHeadModel, RobertaModel

    from data_functions import (get_dataloader, get_yes_no_pad_token_ids,
                                process_boolq, process_sst)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("DEBUG: DEVICE: ", device)

    train_data, train_labels = process_boolq(
        "train", append_answer=True, subset=use_subset
    )
    train_dataloader = get_dataloader(
        train_data, train_labels, "distilgpt2", "left", "left", 128, 16
    )

    val_data, val_labels = process_boolq("validation", subset=use_subset)
    val_dataloader = get_dataloader(
        val_data, val_labels, "distilgpt2", "left", "left", 128, 16
    )

    n, y, p = get_yes_no_pad_token_ids()

    model = GPT2LMHeadModel.from_pretrained("distilgpt2")
    trainer = Trainer(model, "classification", "instruction-tuning", device)

    print("DEBUG: ZERO SHOT RESULTS")
    preds = trainer.inference(val_dataloader, [n, y])
    print("DEBUG: ", Counter(preds))
    print("DEBUG: ZERO SHOT ACCURACY: ", accuracy_score(val_labels, preds))

    # print("DEBUG: INSTRUCTION-TUNING TRAINING")
    # trainer.train(train_dataloader, None, 2)

    # print("DEBUG: INSTRUCTION-TUNED RESULTS")
    # preds = trainer.inference(val_dataloader, [n, y])
    # print("DEBUG: ", Counter(preds))
    # print("DEBUG: INSTRUCTION-TUNED ACCURACY: ", accuracy_score(val_labels, preds))

    model = RobertaModel.from_pretrained("distilroberta-base")
    model.pooler.dense = nn.Linear(model.pooler.dense.in_features, 1)
    model.pooler.activation = nn.Sigmoid()

    print("DEBUG: FINE-TUNING TRAINING")
    trainer = Trainer(model, "classification", "fine-tuning", device)
    trainer.train(train_dataloader, val_dataloader, 2)

    print("DEBUG: FINE-TUNED RESULTS")
    preds = trainer.inference(val_dataloader)
    print("DEBUG: ", Counter(preds))
    print("DEBUG: FINE-TUNED ACCURACY: ", accuracy_score(val_labels, preds))

    train_data, train_labels = process_sst("train", subset=use_subset)
    val_data, val_labels = process_sst("validation", subset=use_subset)
    test_data, test_labels = process_sst("test", subset=use_subset)
    train_dataloader = get_dataloader(
        train_data, train_labels, "distilroberta-base", "left", "left", 128, 16
    )
    val_dataloader = get_dataloader(
        val_data, val_labels, "distilroberta-base", "left", "left", 128, 16
    )
    test_dataloader = get_dataloader(
        test_data, test_labels, "distilroberta-base", "left", "left", 128, 16
    )
    model = RobertaModel.from_pretrained("distilroberta-base")
    model.pooler.dense = nn.Linear(model.pooler.dense.in_features, 1)
    model.pooler.activation = nn.Sigmoid()

    print("DEBUG: REGRESSION TRAINING")
    trainer = Trainer(model, "regression", "fine-tuning", device)
    trainer.train(train_dataloader, val_dataloader, 2)

    print("DEBUG: FINE-TUNED RESULTS")
    preds = trainer.inference(test_dataloader)
    print("DEBUG: MAE: ", mean_absolute_error(test_labels, preds))
