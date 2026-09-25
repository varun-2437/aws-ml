# Amazon ML Challenge 2026 — Business Entity Resolution

A high-performance pipeline for matching business entities across three noisy data sources using TF-IDF blocking, pairwise similarity features, and LightGBM classification.

## Problem

Given business records from 3 sources (S1 = reference, S2/S3 = noisy), determine which S2/S3 records refer to the same real-world business as each S1 record.

**Metric:** Macro-averaged F₀.₅ (precision weighted 2× more than recall)

## Project Structure

```
amazon_ml_challenge/
├── src/
│   ├── __init__.py              # Package marker
│   ├── config.py                # Paths, hyperparameters, constants
│   ├── data_loader.py           # Chunked TSV loading + Parquet caching
│   ├── normalizer.py            # Name/address normalization (NFKC, legal suffixes, Devanagari)
│   ├── evaluation.py            # Macro F₀.₅ scorer with detailed breakdowns
│   ├── blocker.py               # TF-IDF + token + address blocking strategies
│   ├── candidate_generator.py   # Orchestrates blocking by country partition
│   ├── features.py              # 21 pairwise similarity features
│   ├── pair_generator.py        # Labeled pair creation with negative sampling
│   ├── model.py                 # LightGBM training, threshold optimization
│   └── predict.py               # Full test inference → submission TSVs
│
├── dataset/                     # Data files (see dataset/README.md)
│   └── README.md                # Expected dataset structure
│
├── cache/                       # Auto-generated Parquet caches (gitignored)
├── output/                      # Generated submission files (gitignored)
├── utils/
│   └── validate_submission.py   # Official submission validator
│
├── requirements.txt             # Python dependencies
├── .gitignore
└── README.md                    # This file
```

## Setup

```bash
# Clone
git clone https://github.com/varun-2437/aws-ml.git
cd aws-ml

# Install dependencies
pip install -r requirements.txt

# Place dataset files (see dataset/README.md for structure)
# dataset/train/train_source1.tsv, etc.
```

## Pipeline Stages

1. **Data Loading** — Chunked TSV reading with Parquet cache
2. **Normalization** — Legal suffix stripping, accent handling, Devanagari detection
3. **Blocking** — TF-IDF char n-grams + token overlap + address TF-IDF
4. **Feature Engineering** — 21 pairwise similarity features
5. **Model Training** — LightGBM with entity-level train/val split
6. **Threshold Optimization** — Sweep for best macro F₀.₅
7. **Prediction** — Generate `matching_results.tsv` and `candidate_pairs.tsv`

## Key Design Decisions

- **Country partitioning** — Process US → India → France separately to fit in 16GB RAM
- **Multiple blocking strategies** — Union of name TF-IDF, token overlap, and address TF-IDF for >98% recall
- **F₀.₅ optimization** — Threshold tuned directly on the competition metric, not accuracy
- **Devanagari-aware** — Detected and flagged; address TF-IDF catches cross-script matches

## Tech Stack

- Python 3.10+
- pandas, numpy, scikit-learn
- LightGBM
- sparse_dot_topn (fast sparse matrix top-N)
- python-Levenshtein, jellyfish (string similarity)
