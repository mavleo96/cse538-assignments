# Assignment 1

**Stony Brook University**  
**CSE538 - Spring 2025**  
**Assigned:** 02/12/2025  
**Due:** 02/26/2025 11:59pm  

---

## Overview

- **Part I. Tokenizing (40 Points)**
- **Part II. Part-of-Speech Tagging (60 Points)**

#### Objectives:  
- Practice using regular expressions  
- Implement a tokenizer  
- Come to understand the pros and cons of different tokenizers – there is no perfect tokenizer but there are better and worse tokenizers  
- Practice implementing a PyTorch model  
- Become more familiar with logistic regression/maximum entropy classification  
- Experiment with regularization techniques to better understand how they work  

---

## Dataset

The Twitter POS annotated data was introduced in the paper [Part-of-Speech Tagging for Twitter: Annotation, Features, and Experiments](https://aclanthology.org/P11-2008.pdf). Within, you will find many files and you may look at the README.txt for a brief explanation. For the assignment, we will be working with **Daily547**, which is a compilation of 547 unique tweets.

### Part 1: [daily547_tweets.txt](https://drive.google.com/file/d/1OwbJQE2CpsCKuBGLu67PEobHQbCCBWUU/view)
- **Description:** The file contains the 547 tweets, each on a different line. Consider each tweet a "document" that you will tokenize and store.

### Part 2: [daily547_3pos.txt](https://drive.google.com/file/d/1u6iqtwkaEbJuMjk14QTw2X5sYDdZG9yL/view)
- **Description:** Each line in this file contains either (1) a tab-separated token and its POS label, or (2) an empty line between the last and first tokens to denote the end of a sentence.
- While the original version [daily547.conll](https://github.com/brendano/ark-tweet-nlp/blob/master/data/twpos-data-v0.3/daily547.conll) has 25 POS tags (defined in the linked paper), for this assignment we have aggregated them into **3 classes** as follows:  
1. **Noun-related (N)** - originally labeled: N, O, S, ^, Z, L, M, A
2. **Verb (V)** - originally labeled: V
3. **Other (O)** - originally labeled: R ! D P & T X Y # @ ~ U E $ ',' G

## Requirements

- You must use **Python version 3.8** or later with **PyTorch 2.1.1** or later and **Numpy 1.22** or later.
- We will test your code.
- You may do your development in any environment, but runtimes listed will be based on a **2 vCPU system with 1GB memory** and average hard disk drive (an [e2-micro](https://gcloud-compute.com/e2-micro.html) machine available in the free tier for students on GCP).

---

## Python Libraries

Acceptable machine learning, or statistics libraries are listed below (see version numbers for torch and numpy above). Other data science, machine learning, or statistics related libraries are prohibited —- **ask if unsure**. The intention is for you to implement the algorithms we have gone over and problem-solve in order to best understand the concepts of this course and their practical application. **You may not use any pre-existing implementations of such algorithms even if they are provided in the following libraries.**

```python
import random, os, sys, math, csv, re, collections, string
import numpy as np

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

## Part I. Tokenizing (40 Points)

Your objective is to develop and compare **tokenization approaches** for messy social media data.

- **Filename:** `a1_p1_<lastname>_<id>.py`
- **Input:** The file to be tokenized.
    + Example: `a1_p1_lastname_id.py daily547_tweets.txt`
- **Output:** Place checkpoint output into: `a1_p1_<lastname>_<id>_OUTPUT.txt`
- **Mark the checkpoint clearly in your output: `print("Checkpoint X.X:")`**

### 1.1 Word RegEx Tokenizer

Write your own word tokenizer using regular expressions that fulfills the following rules:

1. Retain capitalization.
2. Separate punctuation from words, except for:
    - Abbreviations of capital letters (e.g. `“U.S.A.”`)
    - Contractions (e.g. `“can’t”`)
    - Periods surrounded by integers (e.g. `"5.0"`)<br>
    Examples:
    `“with the fork.”` as `[“with”, “the”, “fork”, “.”]`<br>
    `“Section 5.”` as `[“Section”, “5”, “.”]`
3. Allow for hashtags and @mentions as single words (e.g. `#sarcastic`, `@sbunlp`)

You may handle any situation not covered by these rules however you like.

Place your code in a method named `wordTokenizer`.

```python
def wordTokenizer(sent):
    # input: a single sentence as a string.
    # output: a list of each “word” in the text
    # must use regular expressions

    # <FILL IN>
    return tokens
```

You may create other methods that this method calls.

#### Checkpoint 1.1
Print the output of `wordTokenizer` on the first 5 documents and the very last document.

### 1.2 Spaceless BytePair Tokenizer

Create a BytePair pair encoding tokenizer that does not include whitespace as a valid character. That is, it will not allow a space in any "word." and only use space as indicating a definite split between two words in the corpus.

Example: if the vocabulary was `["i", "n", "a", "b", "in"]` and the corpus was `"abin bi nab ain"` it would be tokenized as: `["a", "b", "in", "b", "i", "n", "a", "b", "a", "in"]` ← note that "i" and "n" was not joined in `"bi nab"` because `"i"` was already separated from `"n"`.

Otherwise, byte pair encoding will proceed as described in the book and slides: Start with a vocabulary of all ascii letters as words. Convert all non-ascii characters into "?".

Run until the vocabulary size reaches 1000.

Place code that learns the vocabulary in a method named `spacelessBPElearn`, and place code that runs the tokenizer in a method named `spacelessBPETokenize`.

```python
def spacelessBPELearn(docs, max_vocabulary=1000):
    # input: docs, a list of strings to be used as the corpus for learning the BPE vocabulary
    # output: final_vocabulary, a set of all members of the learned vocabulary

   return final_vocabulary

def spacelessBPETokenize(text, vocab):
    # input: text, a single string to be word tokenized.
    #        vocab, a set of valid vocabulary words
    # output: words, a list of strings of all word tokens, in order, from the string

   return words
```

You may create other methods that this method calls (for example, training the tokenizer should probably).

#### Checkpoint 1.2
Print the top five most frequent pairs at iterations 0, 1, 10, 100, and 500.<br>
Print the final vocabulary.<br>
Print the tokenization of the first 5 documents and the very last document.

*Code should run in under 2 minutes, although it is possible to run in only a few seconds.*

---

## Part II. Part-of-Speech Tagging (60 Points)

In this part, your objective is to [determine the part of speech for each token in a sentence](https://en.wikipedia.org/wiki/Part-of-speech_tagging) by developing a simple neural network from scratch. Writing code for the training and evaluation of a barebones model using the given function outlines will serve as a stepping stone towards using PyTorch in the next assignment.

- **Filename:** `a1_p2_<lastname>_<id>.py`
- **Input:** Your code should run without any additional arguments.
    + Example: `a1_p2_lastname_id.py daily547_3pos.conll`
- **Output:** Place checkpoint output into: `a1_p2_<lastname>_<id>_OUTPUT.txt`

### 2.0 Loading the Dataset

Use the function `getConllTags` to load the data from <u>daily547.conll</u>. The function will return a list of sentences containing (token, POS tag) pairs.

```python
def getConllTags(filename):
    # input: filename for a conll style parts of speech tagged file
    # output: a list of list of tuples [sent]. representing [[[word1, tag], [word2, tag2]]

    wordTagsPerSent = [[]]
    sentNum = 0
    with open(filename, encoding='utf8') as f:
        for wordtag in f:
            wordtag=wordtag.strip()
            if wordtag: # still reading current sentence
                (word, tag) = wordtag.split("\t")
                wordTagsPerSent[sentNum].append((word,tag))
            else: # new sentence
                wordTagsPerSent.append([])
                sentNum+=1
    return wordTagsPerSent
```

Next, create a dictionary of all unique tokens by iterating over every (token, POS tag) pair, mapping each unique token to an index (e.g. `{‘hello’:0, ‘world’:1}`). Create a similar ID mapping for all POS tags.

We will use these mappings to create one-hot encodings in the next sections.

### 2.1 Lexical Feature Set

Given the target index into tokens in a sentence, return a vector for that token having the following features:

1. Whether the first letter of the target is capitalized (binary feature)
2. The first letter of the target word (i.e. word at index `targetI`): use 257 one-hot values: 0 through 255 can be used simply as ascii values and 256 can be used for anything non-ascii (i.e. when `ord(char) > 255`).
3. The normalized length of the target word as an integer.<br>
Normalize the length as: `min(token_length, 10)/10`
4. One-hot representation of previous word (note: no feature will be 1 if target is first word)<br>
Use `wordToIndex` to make sure the one-hot representation is consistent. `wordToIndex` is a dictionary that, given a word, returns the integer to make 1 in the one-hot representation.<br>
If the word is not in `wordToIndex` then all will be zero.
5. One-hot representation of current word
6. One-hot representation of next word (note: no feature will be 1 if target is last word)

All features should be concatenated into one long flat vector.

```python
def getFeaturesForTarget(tokens, targetI, wordToIndex):
    # input: tokens: a list of tokens in a sentence,
    #        targetI: index for the target token
    #        wordToIndex: dict mapping ‘word’ to an index in the feature list.
    # output: list (or np.array) of k feature values for the given target

    #<FILL IN>

    return featureVector
```

Process each token in the dataset using `getFeaturesForTarget` to obtain a feature matrix (**X**) with the dimensions (num. of all tokens, feature vector length). Then, create ground-truth vector (**y**) with the indices of POS tag IDs using the mapping defined in section 2.0.

Split the dataset (**X** and **y**) into 70% train and 30% dev subsets.

#### Checkpoint 2.1
Print the sum of the first and last 5 individual feature vectors of **X**.

### 2.2 Train Logistic Regression

Implement multiclass logistic regression training using features from 2.1. Use a learning rate and L2 penalty 0.01 with the SGD optimizer and train the model for 100 epochs. You need to use the full training set and not mini batches during forward pass.

Place your code in a method named `trainLogReg`.

```python
def trainLogReg(train_data, dev_data, learning_rate, l2_penalty):
    # input: train/dev_data - contain the features and labels for train/dev splits
    # input: learning_rate, l2_penalty - hyperparameters for model training
    # output: model - the trained pytorch model
    # output: train/dev_losses - a list of train/dev set loss values from each epoch
    # output: train/dev_accuracies - a list of train/dev set accuracy from each epoch

   return model, train_losses, train_accuracies, dev_losses, dev_accuracies
```

#### Checkpoint 2.2
Plot the training and dev set loss and accuracy curves. The plots have epochs as the x-axis and loss/accuracy as the y-axis. Paste the loss curve into your output text file and save it as a pdf.


### 2.3 Hyperparameter Tuning

Improve your model by tuning hyperparameters related to learning rate and regularization. Using grid search for the model over dev set, you will try to find the best learning rates, trying `[0.1, 1, 10]` as well as find the best L2 penalty, trying `[1e-5, 1e-3, 1e-1]`. Find the dev set accuracy over for each configuration.

Place your code in a method named `gridSearch`, which is able to evaluate any model it takes as input:

```python
def gridSearch(train_set, dev_set, learning_rates, l2_penalties):
    # input: learning_rates, l2_penalties - each is a list with hyperparameters to try
    #        train_set - the training set of features and outcomes
    #        dev_set - the dev set of features and outcomes
    # output: model_accuracies - dev set accuracy of the trained model on each hyperparam combination
    #         best_lr, best_l2_penalty - learning rate and L2 penalty combination with highest dev set accuracy

   return model_accuracies, best_lr, best_l2_penalty
```

Next, train the best model using the best hyperparameters from `gridSearch` to obtain its loss and accuracy curves.

#### Checkpoint 2.3
1. Print a table with the dev set accuracy values for all combinations of:<br>
    - columns: l2 penalty
    - rows: learning rate as follows

    | LR \ L2  | 1e-5  | 1e-3  | 1e-1  |
    |----------|-------|-------|-------|
    | 0.1      | x     | x     | x     |
    | 1        | x     | x     | x     |
    | 10       | x     | x     | x     |
    
    Print the combination that worked the best.

2. Include the loss and accuracy curve from the model trained with best hyperparameters.

Your best dev set accuracy should be above 0.75 (i.e. 75%) or else something is off (it's possible for the best to be even a couple points greater than this).

### 2.4 Best Model Inference

To verify how well the model is working, we will test it on some unseen examples.

Tokenize the sentences given below using `wordTokenizer` (section 1.1) and generate the feature vectors for each token using `getFeaturesForTarget` (section 2.1). Use the best model to predict the POS tags.

```python
sampleSentences = \
    ['The horse raced past the barn fell.',
     'For 3 years, we attended S.B.U. in the CS program.',
     'Did you hear Sam tell me to "chill out" yesterday? #rude']
```

#### Checkpoint 2.4
1. For each sample sentence, print the POS tag predicted for each token obtained from `wordTokenizer`.
2. What is your observation about the qualitative performance of the best model?

*Code should complete steps through 2.4 within 7-8 minutes (based on Colab CPU).*