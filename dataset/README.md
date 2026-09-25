# Dataset Directory Structure

> **The dataset files are NOT included in this repository** (they are ~2 GB total).
> Download them from the Amazon ML Challenge 2026 portal and place them here.

## Expected Structure

```
dataset/
├── train/
│   ├── train_source1.tsv        # ~2.2M rows, 136 MB  (S1: reference businesses)
│   ├── train_source2.tsv        # ~5.0M rows, 332 MB  (S2: noisy records)
│   ├── train_source3.tsv        # ~5.3M rows, 339 MB  (S3: noisy records)
│   └── train_ground_truth.tsv   # ~2.2M rows           (S1→S2/S3 match labels)
│
└── test/
    ├── test_source1.tsv         # S1 entities to match
    ├── test_source2.tsv         # S2 candidates
    └── test_source3.tsv         # S3 candidates
```

## File Format

All files are **tab-separated** (TSV) with the following columns:

### Source files (`*_source*.tsv`)
| Column | Type | Description |
|:-------|:-----|:------------|
| `entity_id` | string | Unique ID (e.g., `S1-000001`, `S2-000001`, `S3-000001`) |
| `business_name` | string | Business name (may contain noise, Devanagari, abbreviations) |
| `business_address` | string | Business address (may be missing in ~3% of S2/S3) |
| `country` | string | `US`, `India`, or `France` (France only in test) |

### Ground truth (`train_ground_truth.tsv`)
| Column | Type | Description |
|:-------|:-----|:------------|
| `source1_entity_id` | string | S1 entity ID |
| `matched_entity_ids` | string | Comma-separated S2/S3 entity IDs (empty = singleton) |

## Country Distribution

| Country | Train S1 | Train S2 | Train S3 | Test |
|:--------|:---------|:---------|:---------|:-----|
| US | 1,323,633 (60%) | 3,016,817 | 3,170,056 | Yes |
| India | 883,188 (40%) | 2,017,799 | 2,115,547 | Yes |
| France | — | — | — | Yes (test only) |

## Quick Validation

After placing the files, verify with:
```bash
wc -l dataset/train/*.tsv
# Expected (approx):
#  2206822 train_source1.tsv   (2,206,821 + header)
#  5034617 train_source2.tsv
#  5285604 train_source3.tsv
#  2206822 train_ground_truth.tsv
```
