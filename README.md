# Amazon ML Challenge — Entity Matching System

An NLP-based entity resolution pipeline designed to identify matching entities across a large-scale dataset containing **39M+ records**. The project combines efficient candidate generation techniques with semantic similarity using a fine-tuned SBERT model.

## Overview

Entity matching is the process of determining whether two records refer to the same real-world entity despite variations in names, addresses, formatting, spelling, and other textual attributes.

This project implements a multi-stage pipeline:

**Data Cleaning → Candidate Generation → Similarity Filtering → SBERT Semantic Matching → Entity Resolution**

The pipeline was designed to efficiently process millions of records while reducing the number of candidate pairs that require computationally expensive semantic comparison.

## Key Features

- Processed and cleaned a dataset containing **39M+ records**
- Applied text normalization and preprocessing
- Implemented **frequency-based blocking** for candidate generation
- Used **n-gram matching** to identify potential candidates
- Applied **Jaro-Winkler similarity** for string-based candidate filtering
- Fine-tuned **SBERT (`all-MiniLM-L6-v2`)** for semantic entity matching
- Trained on **600K labeled entity pairs**
- Achieved **98.75% validation accuracy**

## Pipeline

### 1. Data Cleaning

Raw entity information was normalized to reduce variations caused by:

- Case differences
- Special characters
- Extra whitespace
- Formatting inconsistencies
- Variations in names and addresses

The cleaned textual fields were then used for downstream candidate generation.

### 2. Candidate Generation

Comparing every possible pair of records is computationally expensive at this scale.

To reduce the search space, multiple blocking and matching techniques were applied:

- Frequency-based blocking
- N-gram matching
- Sorted-neighborhood style candidate generation
- Jaro-Winkler similarity

These methods reduced the number of candidate pairs passed to the semantic matching stage.

### 3. SBERT Fine-Tuning

A Sentence-BERT model based on:

`all-MiniLM-L6-v2`

was fine-tuned using labeled entity pairs.

The training dataset contained:

- **300K positive pairs**
- **300K negative pairs**
- **600K total training pairs**

The model learned to distinguish whether two entity records referred to the same underlying entity.

### 4. Semantic Matching

After candidate generation, SBERT embeddings were used to capture semantic similarity between entity records.

Similarity scores were then used with a matching threshold to determine whether candidate pairs represented the same entity.

## Dataset Scale

| Dataset | Records | Positive | Negative |
|---|---:|---:|---:|
| Training | 31.1M | 1.26M | 29.88M |
| Validation | 3.91M | 158K | 3.76M |
| Test | 3.90M | 156K | 3.74M |

The model fine-tuning dataset was separately constructed as a balanced set of **600K labeled pairs**.

## Model Performance

The fine-tuned SBERT model achieved:

**98.75% validation accuracy**

The evaluation was performed on held-out entity pairs to measure the model's ability to distinguish matching and non-matching records.

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Sentence Transformers
- SBERT
- Jaro-Winkler Similarity
- N-Gram Matching
- Jupyter Notebook
- Kaggle

## Repository Structure

```text
Amazon-ML-Entity-Matching/
│
├── data-blocking.ipynb
├── training-data-split.ipynb
├── model.py
├── README.md
└── .gitignore
