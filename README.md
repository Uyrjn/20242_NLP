# 20242_NLP
# Aspect-Based Sentiment Analysis (ABSA) with BERT and T5

## Project Overview

This project focuses on Aspect-Based Sentiment Analysis (ABSA) using transformer-based models such as BERT and T5. The goal is to identify aspect terms in sentences and classify their corresponding sentiment polarities (positive, negative, neutral).

We use the **SemEval 2014 Task 4** dataset, which contains restaurant and laptop reviews annotated with aspect terms and sentiment labels.

## Dataset

- **Source:** SemEval 2014 Task 4 – Laptop and Restaurant Reviews
- **Language:** English
- **Annotations:**
  - Aspect terms with their positions in sentences.
  - Sentiment polarity labels (positive, negative, neutral) for each aspect term.

## Model Architecture

- **Aspect Term Extraction:** Treated as a sequence labeling task with BIO tagging, implemented and fine-tuned using BERT and T5.
- **Sentiment Classification:** A separate classification model predicts sentiment polarity for each extracted aspect term.
- Encoder-only models like BERT are primarily used for token-level classification tasks in this project.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Uyrjn/20242_NLP.git
   cd 20242_NLP
