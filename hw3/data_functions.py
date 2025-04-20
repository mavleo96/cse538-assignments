from typing import List, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer


# This function is also used in a1_p2_murugan_116745378.py
def process_boolq(
    split: str, append_answer: bool = False, subset: bool = False
) -> Tuple[List[str], List[int]]:
    """Process the BoolQ dataset"""
    boolq_dataset = load_dataset("google/boolq")
    data = boolq_dataset[split]
    if subset:
        data = data.select(range(1600 if split == "train" else 800))
    if append_answer:
        li = [
            f"{x['passage']}.\n{x['question']}?\n{'yes' if x['answer'] else 'no'}"
            for x in data
        ]
    else:
        li = [f"{x['passage']}.\n{x['question']}?\n" for x in data]
    labels = [1 if x["answer"] else 0 for x in data]
    return li, labels


def process_sst(split: str, subset: bool = False) -> Tuple[List[str], List[float]]:
    """Process the SST dataset"""
    data = load_dataset("stanfordnlp/sst")[split]
    if subset:
        data = data.select(range(100))
    return data["sentence"], data["label"]


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
        torch.tensor(labels),
    )
    dataloader = DataLoader(tensor_data, batch_size=batch_size, shuffle=False)
    return dataloader


def get_yes_no_pad_token_ids() -> Tuple[int, int, int]:
    """Get the token ids for yes, no, and pad token for distilgpt2"""
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    no_token_id = tokenizer.encode("no")[0]
    yes_token_id = tokenizer.encode("yes")[0]
    pad_token_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    return no_token_id, yes_token_id, pad_token_id


if __name__ == "__main__":
    y, n, p = get_yes_no_pad_token_ids()
    print(y, n, p)

    print("--------------------------------")

    data, labels = process_boolq("train", append_answer=True, subset=True)
    print(data[0])
    print(labels[0])

    print("--------------------------------")

    data, labels = process_boolq("train", append_answer=False, subset=True)
    print(data[0])
    print(labels[0])

    print("--------------------------------")

    data, labels = process_boolq("validation", append_answer=False, subset=True)
    print(data[0])
    print(labels[0])

    print("--------------------------------")

    dataloader = get_dataloader(
        data,
        labels,
        "distilgpt2",
        padding_side="left",
        truncation_side="left",
        context_length=128,
        batch_size=8,
    )
    for batch in dataloader:
        print(batch)
        break

    print("--------------------------------")

    data, labels = process_sst("train", subset=True)
    print(data[0])
    print(labels[0])

    print("--------------------------------")

    data, labels = process_sst("validation", subset=True)
    print(data[0])
    print(labels[0])

    print("--------------------------------")

    dataloader = get_dataloader(
        data,
        labels,
        "distilroberta-base",
        padding_side="right",
        truncation_side="right",
        context_length=128,
        batch_size=8,
    )
    for batch in dataloader:
        print(batch)
        break
