# CSE 538 - Natural Language Processing Assignments

This repository contains assignments for the CSE 538 Natural Language Processing course at Stony Brook University by Prof. Andrew Schwartz. The assignments focus on implementing and experimenting with various NLP techniques and models.

## Repository Structure
```
.
├── .gitignore
├── README.md
├── environment_linux.yml
├── environment_mac.yml
├── hw1
│   ├── README.md
│   ├── a1_exp_nb.ipynb
│   ├── a1_p1_murugan_116745378.py
│   ├── a1_p2_murugan_116745378.py
│   ├── data
│   │   ├── daily547_3pos.txt
│   │   └── daily547_tweets.txt
│   └── submission
│       ├── a1_p1_murugan_116745378_OUTPUT.txt
│       ├── a1_p2_murugan_116745378_OUTPUT.pdf
│       └── a1_p2_murugan_116745378_OUTPUT.txt
├── hw2
│   ├── README.md
│   ├── a2_exp_nb.ipynb
│   ├── a2_p1_murugan_116745378.py
│   ├── a2_p2_murugan_116745378.py
│   ├── data
│   │   └── songs.csv
│   └── submission
│       ├── a2_p1_murugan_116745378_OUTPUT.txt
│       └── a2_p2_murugan_116745378_OUTPUT.pdf
└── practice/
```

## Setup Instructions

### Environment Setup

1. Clone the repository:
```bash
git clone [repository-url]
cd cse538-assignments
```

2. Create and activate the Conda environment:

```bash
conda env create -f environment_linux.yml
conda activate cse538
```

## Assignments

### Assignment 1: Text Classification and Word Embeddings
Implementaion of regex and byte pair encoding tokenization techniques and POS tagging models using Twitter data, focusing on text preprocessing and classification tasks.

### Assignment 2: Language Models and Text Generation
Development of N-gram and RNN-based language models for text generation, using song lyrics to compare different modeling approaches.

> Note: Detailed requirements, datasets, and instructions for each assignment can be found in their respective directories (`hwX/README.md`).

## Dependencies

The project uses Python 3.12 and includes the following main dependencies:
- PyTorch
- Transformers
- NumPy
- Matplotlib
- scikit-learn
- Jupyter Notebook

All dependencies are specified in the environment YAML files.

<!-- ## Results

Assignment results and outputs are stored in the respective `results/` directories within each assignment folder. -->

## License

All rights reserved. This project is for educational purposes only and part of the CSE 538 coursework.