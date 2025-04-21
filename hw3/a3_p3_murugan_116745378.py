#! /usr/bin/env python3

import argparse
import os
import random

import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate
from tqdm import tqdm
from transformers import pipeline

random.seed(0)
torch.manual_seed(0)


# ==========================
#     Data Functions
# ==========================


def get_unique_ctx_examples(squad, n=500):
    context2idx = {}
    for i, entry in enumerate(squad["validation"]):
        if not entry["context"] in context2idx:
            context2idx[entry["context"]] = []
        context2idx[entry["context"]].append(i)

    queries, contexts, answers = [], [], []
    for k, v in context2idx.items():
        idx = v[0]
        queries.append(squad["validation"][idx]["question"])
        contexts.append(squad["validation"][idx]["context"])
        answers.append(squad["validation"][idx]["answers"])
        if len(queries) == n:
            break
    return queries, contexts, answers


# ==========================
#     RAG Functions
# ==========================


def retrieve(contexts, embeddings, query):
    """Retrieves the context with the highest cosine similarity to the query"""

    # sentence_model is global variable as function signature does not include it
    query_embedding = sentence_model.encode(query).reshape(1, -1)
    similarities = cosine_similarity(embeddings, query_embedding)
    idx = similarities.argmax()
    ret_context = contexts[idx]

    return idx, ret_context


SYSTEM_PROMPT = """
You are a helpful AI assistant.
Provide one Answer ONLY to the following query based on the context provided below.
Do not generate or answer any other questions.
Do not make up or infer any information that is not directly stated in the context.
Provide an answer that is precise and concise in one word or as few words as possible.
If you cannot find the answer in the context, return "NA".

Context:
{ret_context}
"""

QUERY_PROMPT = """
Query:
{query}
"""


def generate_response(model, query, ret_context):
    """Generates a response to the query using the retrieved context"""

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT.format(ret_context=ret_context),
        },
        {
            "role": "user",
            "content": QUERY_PROMPT.format(query=query),
        },
    ]

    generation_args = {
        "max_new_tokens": 100,
        "return_full_text": False,
        "do_sample": False,
    }
    response = model(messages, **generation_args)

    return response


# ==========================
#     Main Function
# ==========================


def main() -> None:
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="script to run cse538 assignment 3 part 3"
    )
    parser.add_argument(
        "--model_id", type=str, default="meta-llama/Llama-3.2-3B-Instruct"
    )
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--file_prefix", type=str, default="a3_p3_murugan_116745378")
    args = parser.parse_args()
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")
    if args.model_id in {
        "microsoft/Phi-3-mini-4k-instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
    }:
        print(f"Using model: {args.model_id}")
    else:
        raise ValueError(f"Unknown model: {args.model_id}")

    # Create and open output file
    print("Creating output file...")
    os.makedirs(args.save_dir, exist_ok=True)
    outfile = open(f"{args.save_dir}/{args.file_prefix}_OUTPUT.txt", "w")

    print("Loading dataset...")
    squad = load_dataset("squad")

    print("Loading sentence model...")
    global sentence_model
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Loading examples and embeddings...")
    queries, contexts, answers = get_unique_ctx_examples(squad)
    context_embeddings = sentence_model.encode(contexts)

    print("Retrieving context...")
    outfile.write("Checkpoint 3.1:\n")
    correct_retrieval_ids, incorrect_retrieval_ids = [], []
    for qid, query in tqdm(
        enumerate(queries), desc="Retrieving context", total=len(queries)
    ):
        cid, _ = retrieve(contexts, context_embeddings, query)
        # Saving query and retrieved context ids for downstream use
        if cid == qid:
            correct_retrieval_ids.append((qid, cid))
        else:
            incorrect_retrieval_ids.append((qid, cid))

    outfile.write(
        f"Retrieval accuracy: {len(correct_retrieval_ids)}/{len(queries)} = {len(correct_retrieval_ids) / len(queries)}\n\n"
    )

    print("Creating pipeline...")
    pipe = pipeline(
        "text-generation",
        model=args.model_id,
        device=device,
        torch_dtype=torch.float16,
    )

    print("Generating responses...")
    outfile.write("Checkpoint 3.2:\n")
    # Randomly sample 5 correct and 5 incorrect retrievals
    correct_ids = random.sample(correct_retrieval_ids, 5)
    incorrect_ids = random.sample(incorrect_retrieval_ids, 5)
    outputs = []
    for qid, cid in tqdm(correct_ids + incorrect_ids, desc="Generating responses"):
        response = generate_response(pipe, queries[qid], contexts[cid])
        row = [qid, cid, response[0]["generated_text"], answers[qid]["text"][0]]
        outputs.append(row)
    outfile.write(
        tabulate(
            outputs,
            headers=["query_id", "context_id", "generated", "actual"],
            tablefmt="pretty",
        )
    )

    # Close output file
    print("Closing output file...")
    outfile.close()

    return None


if __name__ == "__main__":
    main()
