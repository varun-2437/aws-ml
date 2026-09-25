# Progress Tracker

## ✅ Done

### Phase 1 — Data Loading & Exploration
- Mapped directory structure and file sizes (~2 GB, ~12.5M records across 3 sources)
- Built chunked TSV reader with Parquet caching (`data_loader.py`)
- Analyzed ground truth: 5.6% singletons, median 3 matches/entity, max 11
- Verified country distribution: Train = US (60%) + India (40%), Test = US + India + France
- Detected ~3-5% Devanagari noise in S2/S3 names, ~3.3% missing addresses
- Inspected official validator (`utils/validate_submission.py`)

### Phase 2 — Normalization
- Built normalizer with NFKC unicode handling (`normalizer.py`)
- Legal suffix stripping: US (Inc, LLC, Corp), India (Pvt Ltd, LLP), France (SARL, SAS, SA)
- Address abbreviation normalization (Road→Rd, Street→St, Boulevard→Blvd, etc.)
- Devanagari script detection (flagged, not destroyed)
- French accent preservation (fixed NFKD→NFKC bug)
- Domain extension removal (.com, .in, etc.)
- Noise character stripping (<<, --, etc.)

### Phase 3 — Blocking (code ready, not yet run on full data)
- TF-IDF character n-gram blocking (top-50 per S1 entity)
- Token-based inverted index blocking (≥2 shared tokens)
- Address TF-IDF blocking (top-20, catches Devanagari mismatches)
- Address number + city token blocking
- Union with cap at 100 candidates/entity
- Blocking recall measurement utility

### Phase 4 — Feature Engineering (code ready)
- 21 pairwise similarity features:
  - Name: Levenshtein, Jaro-Winkler, Jaccard (token + char3), containment, sorted-token sim, length ratio, common prefix ratio, first word match, legal suffix match, token count diff
  - Address: Levenshtein, Jaccard token, number match, PIN/ZIP match, length ratio, missing flag
  - Meta: source type (S2/S3), Devanagari mismatch, combined score, name-address agreement

### Phase 5 — Model (code ready)
- LightGBM binary classifier with auto class weighting
- Entity-level train/val split (no leakage)
- Threshold sweep optimizing macro F₀.₅

### Phase 6 — Evaluation (code ready)
- Macro-averaged F₀.₅ scorer matching competition definition
- Detailed breakdown: per-country, singletons, precision, recall
- All self-tests passing

### Phase 7 — Prediction Pipeline (code ready)
- End-to-end: load → normalize → block → features → predict → write TSVs
- Country-partitioned processing for memory management
- Output: `matching_results.tsv` + `candidate_pairs.tsv`

### Audit & Testing
- Fixed 5 bugs (int8 overflow, NFKD accents, unscaled feature, getattr fragility, unused import)
- Added 3 improvements (address TF-IDF blocking, token count diff feature, JW combined score)
- Integration test on 5K sample: 100% blocking recall, F₀.₅ = 0.97
- All 10 modules import cleanly, zero NaN/Inf in features

### Repo Setup
- Pushed to GitHub: https://github.com/varun-2437/aws-ml.git
- Dataset gitignored with structure documented in `dataset/README.md`

---

## 🔲 To Do

### Phase 3b — Run Blocking on Full Data
- Run `candidate_generator.py` on full 2.2M S1 × 10.3M S2+S3
- Measure blocking recall (target: ≥98%)
- Tune TF-IDF top-K and token overlap threshold if recall is low
- Estimated time: ~30-60 min

### Phase 4b — Generate Training Pairs on Full Data
- Build feature matrix from full blocking candidates + ground truth
- Negative sampling (50 negatives per entity)
- Estimated: ~100M+ candidate pairs → feature computation (~1-2 hours)

### Phase 5b — Train Final Model
- Train LightGBM on full feature matrix
- Optimize threshold on validation set
- Analyze feature importance, check for overfitting

### Phase 6b — Evaluate on Full Validation Set
- Report macro F₀.₅ overall and per-country (US, India)
- Analyze error cases: singletons, partial matches, false merges

### Phase 7b — Generate Test Predictions
- Run full pipeline on test data (includes France — unseen country)
- Write submission files to `output/`
- Validate with `utils/validate_submission.py`

### Phase 8 — Iteration & Improvements (if time permits)
- Add more features: phonetic matching (Soundex/Metaphone), fuzzy address parsing
- Try Devanagari transliteration for cross-script matching
- Ensemble multiple thresholds per country
- Tune LightGBM hyperparameters (Bayesian optimization)
- Add hard negative mining (re-train on model's worst false positives)

### Phase 9 — Final Submission
- Validate both output files pass official validator
- Double-check row counts match test S1 count
- Submit to competition portal
