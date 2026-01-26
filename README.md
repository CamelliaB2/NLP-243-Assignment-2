# Slot Tagging of Natural Language Utterances (Sequence Labeling)

This project focuses on **slot tagging**, a classic **sequence labeling** problem in NLP. Given a natural-language utterance about movies or people, the task is to assign an **IOB slot tag** to each token in the sentence. The project explores multiple neural architectures and systematically evaluates how model choice, feature extraction, and hyperparameters affect performance.

---

## Table of Contents
1. [Problem Overview](#problem-overview)
2. [Dataset Description](#dataset-description)
3. [Task Formulation](#task-formulation)
4. [Slot Tagging with IOB Labels](#slot-tagging-with-iob-labels)
5. [Model Architectures Explored](#model-architectures-explored)
6. [Input Representation and Tokenization](#input-representation-and-tokenization)
7. [Training Setup](#training-setup)
8. [Hyperparameter Optimization](#hyperparameter-optimization)
9. [Experiments and Results](#experiments-and-results)
10. [Key Observations](#key-observations)
11. [Limitations](#limitations)
12. [Future Improvements](#future-improvements)
13. [Notes to Future Me](#notes-to-future-me)

---

## Problem Overview

The goal of this project is to perform **slot tagging** on natural-language utterances. Each utterance is a short query asking for information about a movie or person, and the task is to identify and label relevant spans (slots) within the sentence.

Example:
- Input: *"who plays luke on star wars new hope"*
- Utterance: *"who plays luke in star wars new hope"*
- Output:
  - `Luke` → `B_char`
  - `Star Wars New Hope` → `B_movie I_movie I_movie I_movie`

This task is commonly used in:
- Dialogue systems
- Information extraction
- Question answering pipelines

---

## Dataset Description

The dataset is based on the **Freebase film schema**, the same domain used in HW1.

### Training Set (`hw2_train.csv`)
- 2,313 utterances
- Columns:
  - `ID`
  - `utterance`
  - `IOB slot tags`

### Test Set (`hw2_test.csv`)
- 981 utterances
- Columns:
  - `ID`
  - `utterance`

### Slot Tag Vocabulary
- Total of **27 tags**
- Includes:
  - `B_movie`, `I_movie`
  - `B_actor`, `I_actor`
  - `B_director`, `I_director`
  - `B_release_year`, `I_release_year`
  - `O` (non-slot tokens)

---

## Task Formulation

This is a **sequence labeling** problem:
- Input: sequence of tokens
- Output: sequence of slot labels (same length as input)

Unlike classification:
- Predictions are made **per token**
- Context matters across the entire sequence

The loss function must handle:
- Multi-class token-level prediction
- Padding tokens (ignored during loss computation)

---

## Slot Tagging with IOB Labels

The project uses the **IOB tagging scheme**:
- `B-<slot>`: beginning of a slot
- `I-<slot>`: inside a slot
- `O`: token not part of any slot

Why IOB matters:
- Allows multi-token spans
- Preserves slot boundaries
- Standard in NER and slot tagging tasks

---

## Model Architectures Explored

Because slot tagging is inherently sequential, the project focuses on **recurrent models**, gradually increasing architectural complexity.

### Baseline Models
- **LSTM**
- **GRU**
- **Elman RNN**

These models capture temporal dependencies across tokens.

### Extended Architectures
- **RNN + CNN**
  - CNN extracts local n-gram features
  - RNN captures sequence-level context
- **Bidirectional variants**
  - Capture both left and right context

---

## Input Representation and Tokenization

### Tokenization
- **BERT tokenizer** used across experiments
- Breaks words into subword units
- Helps with:
  - Unknown words
  - Rare entity names

### Embeddings
- Token embeddings fed into RNN-based models
- Embedding dimensions tuned via hyperparameter optimization

---

## Training Setup

### Data Split
- 80% training
- 20% validation

### Loss Function
- **CrossEntropyLoss**
- Padding tokens ignored using `ignore_index`

### Optimizers Tested
- Adam
- AdamW
- NAdam
- Adagrad (tested, but less effective)

### Hyperparameter Optimization
- **Optuna** used for automated tuning
- Parameters explored:
  - Embedding dimension
  - Hidden dimension
  - Learning rate
  - Dropout
  - Number of CNN filters

---

## Experiments and Results

### Baseline Findings
- LSTM, GRU, and RNN performed **similarly**
- GRU slightly more stable in training

### CNN Augmentation
- Adding CNN layers significantly improved F1 scores
- CNNs help capture local patterns useful for slot boundaries

### Bidirectionality
- Bidirectional RNNs consistently outperformed unidirectional variants
- Provided the largest single performance gain

### Best Performing Model
- **GRU + CNN**
- Hyperparameters tuned via Optuna
- Best Kaggle submission F1 ≈ **0.66**

---

## Key Observations and Lessons Learned

- Architecture complexity helps, but only when aligned with the task
- CNN + RNN is a strong combination for sequence tagging
- Bidirectional context matters more than model choice
- Lower learning rates stabilize training significantly
- Validation loss can increase even when F1 improves (likely mild overfitting)
- Hyperparameter tuning had a larger impact than swapping optimizers

---

## Limitations

- No CRF layer (which is common for slot tagging)
- Evaluation focused mostly on F1; per-slot metrics could be expanded
- Some overfitting signs in deeper models
- No cross-validation (single split)

---

## Potential Improvements

If revisiting this project:
1. Add a **CRF decoding layer**
2. Perform **per-slot precision/recall analysis**
3. Introduce **early stopping**
4. Use **transformer-only** architectures for comparison
5. Apply **label smoothing**
6. Experiment with span-based modeling

---

## Notes to Future Me

This project was less about absolute performance and more about:
- Understanding sequence labeling dynamics
- Seeing how architectural components interact
- Learning when added complexity actually helps

The most important insight:
> **Bidirectional context + good feature extraction mattered more than optimizer or depth.**
