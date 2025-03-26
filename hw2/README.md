# Assignment 2

**Stony Brook University**  
**CSE538 - Spring 2025**  
**Assigned:** 03/06/2025
**Due:** 03/26/2025 11:59pm

---

## Overview

- **Part I. NGram LM (50 Points)**
    - 1.1 Tokenize (10 points)
    - 1.2 Smoothed Trigram Language Model (30 points)
    - 1.3 Perplexity of TrigramLM (10 points)
- **Part II. RNN LM (50 Points)**
    - 2.1 Preparing the Dataset (10 points)
    - 2.2 Recurrent NN-based Language Model (10 points)
    - 2.3 Train RecurrentLM (20 points)
    - 2.4 Autoregressive Lyric Generation (10 points)
    - Extra Credit (10 points)

#### Objectives:
- Work with a pre-trained BPE tokenizer
- Implement two different language modeling approaches
- Evaluate the language models
- Use a language model to generate content

---

## Dataset

In all parts of this assignment, you will use the same dataset covering task domains of language modeling and content generation.

### Taylor Swift Lyrics Dataset [songs.csv](https://github.com/shaynak/taylor-swift-lyrics/blob/main/songs.csv)
- **Description:** A dataset comprising all of Taylor Swift's songs. The CSV file contains three columns - Title, Album, and Lyrics. The lyrics are in plain ASCII text, with the exception of a few accented characters in words like café, rosé, etc.

## Requirements

- You must use **Python version 3.8** or later with **PyTorch 2.1.1** or later and **Numpy 1.22** or later.
- We will test your code.
- You may do your development in any environment, but runtimes listed will be based on a **2 vCPU system with 1GB memory** and average hard disk drive (an [e2-micro](https://gcloud-compute.com/e2-micro.html) machine available in the free tier for students on GCP).

### Colab Policy  
You are welcome to test your code in Colab to utilize the GPU or TPU available there by [uploading your code to drive and then importing it into Colab](https://colab.research.google.com/drive/1YppUP29n7S7w5rZahQeOCE2WVvTCdqys). However, **it is suggested you develop your code with a full code editor (e.g., VS Code, vim, emacs, PyCharm) rather than in Colab or any notebook.** Notebooks encourage a segmented style of coding, which often leads to bugs in AI systems. It is better to use modular and good object-oriented design within Python code files directly. Notebooks are useful for trying out brief segments of code.

---

## Python Libraries

Acceptable machine learning, or statistics libraries are listed below (see version numbers for torch and numpy above). Other data science, machine learning, or statistics related libraries are prohibited —- **ask if unsure**. The intention is for you to implement the algorithms we have gone over and problem-solve in order to best understand the concepts of this course and their practical application. **You may not use any pre-existing implementations of such algorithms even if they are provided in the following libraries.**

```python
import random, os, sys, math, csv, re, collections, string
import numpy as np
import csv

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import heapq
import matplotlib
```

(Note: Do not use `pandas`. It is not included above on purpose.)

---

## Code-Assistance Policy

See syllabus for code-assistance policy.

**Copying code from another student (current or previous), person, or any non-open source is prohibited.** This will result in at least a zero on the assignment and a report to the graduate program director with possibility for more consequences.**Students are equally responsible for making sure their code is not accessible to any other student** as they are for not copying another's code. Please see the syllabus for additional policies. A word to the wise: Multiple subparts of this assignment have never been given before. Take the time to figure them out on your own, seeking only conceptual help (or talking to TAs/instructor at office hours). This is not only less selfish to your classmates, but it will take you further in life.

---

## Part I: NGram LM (50 Points)

Your objective is to implement a basic n-gram language model.

- **Filename:** `a2_p1_<lastname>_<id>.py`
- **Input:** Your code should run without any _additional_ arguments.
    - Example: `python a2_p1_lastname_id.py songs.csv`
- **Output:** Place checkpoint output into: `a2_p1_<lastname>_<id>_OUTPUT.txt`

### 1.1 Tokenize

All components of this assignment will use a pre-trained BPE tokenizer with document start and stop tokens. In particular, parts 1 and 2 will use the GPT2 Tokenizer.

- Start by initializing **[GPT2TokenizerFast](https://huggingface.co/docs/transformers/en/model_doc/gpt2#transformers.GPT2TokenizerFast)**, which is a BPE tokenizer. The "fast" version is quick in tokenizing batches of text sequences. It initially contains `<|endoftext|>` as the Beginning of Sentence (BOS), End of Sentence (EOS), and unknown token.
- Add a pad token with value `<|endoftext|>`.
- Change the BOS and EOS in the tokenizer’s special token map to `<s>` and `</s>` respectively.

**Side Note:** GPT2 does not use start and stop tokens, and you must add them yourself. We are using GPT2 tokenizer here so you can compare probabilistic LM to the output of RNN-based LM in part 2. However, if you were to want to use a tokenizer that has sentence start and stop tokens, like roberta-base, then you would get them with:
```python
tokenizer = PreTrainedTokenizerFast.from_pretrained('roberta-base')

words = tokenizer.convert_ids_to_tokens(tokenizer.encode("When did SBU open?"))
```

Load the dataset from [songs.csv](https://github.com/shaynak/taylor-swift-lyrics/blob/main/songs.csv), keeping all except the last 5 songs as the training set. We will use pieces of the last 5 songs for out-of-sample testing.

Note: You may use the following code to read the dataset (it excludes the header and last 5 songs) for both parts.

```python
with open('songs.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)[1:-5]
```


#### Checkpoint 1.1
Mark the checkpoint clearly in your output: `print("\nCheckpoint 1.1:")`

Print the token list for the "Lyrics" column of the first and last rows of the training set.
```
first: ['This', 'Ġis', 'Ġa', 'Ġnewer', 'Ġtest', 'Ġof', 'Ġtoken', 'izing', 'Ġfrom', 'ĠSt', 'ony', 'ĠBrook', 'ĠUniversity', '.']
last: ['S', 'ton', 'y', 'ĠBrook', 'ĠUniversity', 'Ġwas', 'Ġfounded', 'Ġand', 'Ġfirst', 'Ġheld', 'Ġclasses', 'Ġin', 'Ġ1967', '.', 'Ġ', 'ĠWhen', 'Ġdid', 'ĠSB', 'U', 'Ġopen', '?']
```
Note: Above is an example output, not the answer. Remember that Ġ is the space marker

### 1.2 Smoothed Trigram Language Model

Make a class, `TrigramLM`, that creates and trains a trigram language model. The LM should be trained on all songs from the training set. In particular, `TrigramLM` must have the following instance methods:

#### **Methods:**
1. `trigramLM.train(datasets)`: Stores data necessary to be able to compute smoothed trigram and unigram probabilities (see more specifics below). Does not need to return anything.
    - Dataset is a collection of songs. These can be in any format that suits your approach.
2. `trigramLM.nextProb(history_toks, next_toks)`: Returns the probability of tokens in `next_toks`, given `history_toks`.
    - `history_toks` is a list of previous tokens. Make sure this works for lists of any size, including 0 (i.e. no history: unigram model) to a size > 2 in which case it just uses the final 2 history tokens since it is a trigram model (this will be helpful for future work).
    - `next_toks` is a set of possible next tokens to return probabilities for.

**Unlike assignment 1 you may add additional optional arguments but you must have the above arguments.** The LM must have the following properties:
1. The vocabulary must be the complete vocabulary of the tokenizer.
2. It must utilize an "OOV" token when a word is not available or an ngram count is not available.
3. You must use document start and stop tokens.
4. During training it must store **what is necessary for computing** add-one smoothed trigram and unigram probabilities.

#### Checkpoint 1.2
Print the `nextProb` for the following histories and candidate next tokens:

1. `history_toks=['<s>', 'Are', 'Ġwe']`
`next_toks=['Ġout', 'Ġin', 'Ġto', 'Ġpretending', 'Ġonly']`
2. `history_toks=['And', 'ĠI']`
`next_toks=['Ġwas', "'m", 'Ġstood', 'Ġknow', 'Ġscream', 'Ġpromise']`

### 1.3 Perplexity of TrigramLM (10 points)

Perplexity is a metric to measure how "surprised" a language model is when it sees a test sample. Place your code in `get_perplexity(probs)` to compute this metric as the inverse of the geometric mean over the probabilities of target tokens.

```python
def get_perplexity(probs):
    # input: probs: a list containing probabilities of the target token for each index of input
    # output: perplexity: a single float number

    # <FILL IN>
    return perplexity
```

#### Checkpoint 1.3  
Print the perplexity for the following cases:

1. `['And', 'Ġyou', 'Ġgotta', 'Ġlive', 'Ġwith', 'Ġthe', 'Ġbad', 'Ġblood', 'Ġnow']`
2. `['Sit', 'Ġquiet', 'Ġby', 'Ġmy', 'Ġside', 'Ġin', 'Ġthe', 'Ġshade']`
3. `['And', 'ĠI', "'m", 'Ġnot', 'Ġeven', 'Ġsorry', ',', 'Ġnights', 'Ġare', 'Ġso', 'Ġstar', 'ry']`
4. `['You', 'Ġmake', 'Ġme', 'Ġcraz', 'ier', ',', 'Ġcraz', 'ier', ',', 'Ġcraz', 'ier', ',', 'Ġoh']`
5. `['When', 'Ġtime', 'Ġstood', 'Ġstill', 'Ġand', 'ĠI', 'Ġhad', 'Ġyou']`

What are your observations about these results? Are the values similar or different? What is one major reason for this? (2-4 lines).  

*Code should run in under 3 minutes, although it is possible to run in less than 1 minute.*

---

## Part II. RNN LM (50 Points)

In this part, your objective is to implement the language modeling task using a recurrent neural network.

- **Filename:** `a2_p2_<lastname>_<id>.py`
- **Input:** Your code should run without any additional arguments
    + Example: `a2_p2_lastname_id.py songs.csv`
- **Output:** Place checkpoint output into: `a2_p2_<lastname>_<id>_OUTPUT.pdf`


### 2.1 Preparing the dataset

Now fill in the `chunk_tokens` function below. The function should split a larger token sequence into multiple, smaller sequences of size `chunk_len - 2`. Append the BOS token id at the start and EOS token id at the end of the chunk. If the number of tokens are less than chunk_size, then append pad tokens after EOS.

Same sequence length enables batch processing, which saves time during model training.

```python
def chunk_tokens(tokens, start_token_id, end_token_id, pad_token_id, chunk_len=128):
    # input: tokens: a list containing token ids
    #        start_token_id, end_token_id, pad_token_id: special token ids from the tokenizer
    #        chunk_len: the length of output sequences
    # output: chunks: torch.tensor of sequences of shape (#chunks_in_song, chunk_len)
    #<FILL IN>
    return chunks
```

Load [songs.csv](https://github.com/shaynak/taylor-swift-lyrics/blob/main/songs.csv) excluding the last 5 songs. Process each row in the training set's "Lyrics" column to obtain a tensor of shape (#chunks_in_song, chunk_len):
1. Remove section markers such as [Bridge], [Chorus], etc. by using the regex pattern `r'\n\[[\x20-\x7f]+\]'` to replace it with an empty string.
2. Tokenize current row’s lyrics to get a list of token ids.
3. Call `chunk_tokens` on the list of token ids using `chunk_len = 64`.

Stack each row’s resultant tensor into a single tensor of shape (#all_chunks, chunk_len). Use this to create **X** and **y** for self-supervised learning. **X** should contain all but the last column, whereas **y** excludes the first column. In other words, a column in y gives the next token for corresponding current token in X.

Use `torch.utils.data.TensorDataset` and `torch.utils.data.DataLoader` with  batch size of 32 and `drop_last=True`. It makes batching the dataset more convenient to manage.

#### Checkpoint 2.1
Print the chunked tensors for the song "Enchanted (Taylor's Version)" from the album "Speak Now (Taylor's Version)".

### 2.2 Recurrent NN-based Language Model

Place in the code for `RecurrentLM`, which is a GRU-based language model.
1. `__init__` – takes as input the dimensions for various layers and initializes them.
2. `forward` – accepts a batched sequence of token IDs. Process the input **x** in the order of layers Embedding -> GRU-> Layer norm -> Fully-connected.
3. `stepwise_forward` – complete this in Part 2.4

```python
class RecurrentLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, rnn_hidden_dim):
        super().__init__()
        # <FILL IN> - An embedding layer to convert token IDs into vectors of size embed_dim
        # <FILL IN> - An GRU layer with rnn_hidden_dim sized state vectors. Use batch_first=True
        # <FILL IN> - Layer normalization for RNN's outputs
        # <FILL IN> - Projection from rnn_hidden_size to vocab_size

    def forward(self, x):
        # input: x: tensor of shape (batch_size, seq_len)
        # output: logits: output of the model.
        #         hidden_state: hidden state of GRU after processing x (sequence of tokens)
        # <FILL IN>

        return logits, hidden_state

    # def stepwise_forward(self, x, prev_hidden_state):
        #input: x: tensor of shape (seq_len)
        #       hidden_state: hidden state of GRU after processing x (single token)
        #<FILL IN at Part 2.4>

        # return logits, hidden_state
```

#### Checkpoint 2.2
Print the shape of **logits** and **hidden_state** tensors in terms of `batch_size`, `chunk_len`, `vocab_size` and `rnn_hidden_dim`.


### 2.3 Train RecurrentLM

Initialize `RecurrentLM` with:
1. vocab_size as size of the tokenizer’s vocabulary (including special tokens)
2. `embed_dim = 64`
3. `rnn_hidden_dim = 1024`

Train the model on GPU using mini batch gradient descent, passing features from 2.1 with batch size of **32**. Use a learning rate **0.0007** with the Adam optimizer. Train the model for **15** epochs.

Next-token prediction is also a classification problem, which allows us to use Cross-entropy loss. Transform the logits and labels such that their dimensions are (#examples, #classes). Now we can directly use `nn.CrossEntropyLoss`, but before that, there is one more detail to take care of!

In Part 2.1, pad tokens were appended to sequences shorter than `chunk_len`. These should be handled so that they don’t contribute to a model’s predictions and loss calculations. Filter out the outputs and labels corresponding to indices where the input was a pad token ID. Now, compute the loss on the remaining set of inputs, outputs, and labels.

Place your code in a method named `trainLM`.

```python
def trainLM(model, data, pad_token_id, learning_rate, device):
    # input: model - instance of RecurrentLM to be trained
    #        data - contains X and y as defined in 2.1
    #        pad_token_id - tokenizer’s pad token ID for filtering out pad tokens
    #        learning_rate
    #        device - whether to train model on CPU (="cpu") or GPU (="cuda")
    # output: losses - a list of loss values on the train data from each epoch
    #<FILL IN>

   return losses
```

#### Checkpoint 2.3
1. Plot the training set loss curves. It should have loss on the y-axis and epochs on the x-axis. Paste the loss curve into your output file and save it as a pdf.
2. Compute the perplexity of the model on the samples (you can use `get_perplexity` from Part 1.3)
    - `"And you gotta live with the bad blood now"`
    - `"Sit quiet by my side in the shade"`
    - `"And I'm not even sorry, nights are so starry"`
    - `"You make me crazier, crazier, crazier, oh"`
    - `"When time stood still and I had you"`
3. Compare the perplexity scores with Part 1.3. How does the RNN-based LM perform in comparison? Provide a brief reason why it is or isn't better.

### 2.4 Autoregressive Lyric Generation

Revisit `RecurrentLM` class definition and place your code for `stepwise_forward`. This function accepts a single token ID as input and the hidden state from the previous step to predict the next token. This method is mostly similar to forward, except that now you have to pass the hidden state to the GRU.

Next, place your code in the `generate` method which should
1. Tokenize the `start_phrase`. Since this could be one or more tokens, process it with RecurrentLM.forward to get the logits and hidden state. Pick the logit with the highest probability as the next token and append its ID to `generated_tokens`.
2. Pass this token along with the hidden state to `RecurrentLM.stepwise_foward`. Like the previous step, capture the highest probability logit as the next token. Repeat this process until either `max_len` number of tokens are generated (or) the generated token is EOS/pad.

Use a `max_len = 64`.

```python
def generate(model, tokenizer, start_phrase, max_len, device):
    # input: model - trained instance of RecurrentLM
    #        tokenizer
    #        start_phrase - string containing input word(s)
    #        max_len - max number of tokens to generate
    #        device - whether to inference model on CPU (="cpu") or GPU (="cuda")
    # output: generated_tokens - list of generated token IDs
    #<FILL IN>

   return generated_tokens
```

#### Checkpoint 2.4
Print the content generated with the following start phrases in plain text (not token IDs!)
1. `"<s>Are we"`
2. `"<s>Like we're made of starlight, starlight"`
3. Try your own start phrase here


#### Extra Credit
1. Modify `generate` to sample the next token from the probability distribution over all logits instead of taking the highest logit. Generate content with start phrases from Checkpoint 2.4 and print it.
2. Which approach appears to generate more original/unique content and why?

*Code should complete steps through 2.4 within 5-7 minutes (based on Colab GPU; note see colab policy).*

---

## Submission

Submit the following 4 files containing the output of your code as well as your code itself. Please use brightspace:
1. `a2_p1_lastname_id.py`
2. `a2_p1_lastname_id_OUTPUT.txt`
3. `a2_p2_lastname_id.py`
4. `a2_p2_lastname_id_OUTPUT.pdf`

**Please do not upload a zip or a notebook file. Double-check that your files are there and correct after uploading and make sure to submit.**  Uploading files that are zips or any other type than .py, .pdf, or .txt files will result in the submission being considered invalid. Partially uploaded files or non-submitted files will count as unsubmitted.

If submitting multiple times, only the last submission will be graded. If it is late, the penalty will apply even if earlier submissions were made.

**Questions:** Please post questions to the course forum.

---

## Additional References
