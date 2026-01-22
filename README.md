# Slot Tagging of Natural Language Utterances

This project focuses on **slot tagging (sequence labeling)** for natural language queries using deep learning. The task is to identify and label semantic slots within film-related utterances using the IOB tagging scheme. Multiple neural architectures were explored, with an emphasis on understanding how architectural choices and hyperparameters affect performance.

---

## Table of Contents
1. [Problem Overview](#problem-overview)
2. [Dataset Description](#dataset-description)
3. [Task Formulation](#task-formulation)
4. [IOB Tagging Scheme](#iob-tagging-scheme)
5. [Models Explored](#models-explored)
6. [Input Representation](#input-representation)
7. [Training Setup](#training-setup)
8. [Hyperparameter Optimization](#hyperparameter-optimization)
9. [Experiments and Results](#experiments-and-results)
10. [Key Observations](#key-observations)
11. [Limitations](#limitations)
12. [Future Improvements](#future-improvements)
13. [Notes to Future Me](#notes-to-future-me)

---

## Problem Overview

The goal of this project is **slot tagging**, a form of sequence labeling in NLP. Given a natural-language utterance, the model must assign a semantic tag to **each token** in the sequence.

Example:
- Input: *"who plays luke on star wars new hope"*
- Output:
