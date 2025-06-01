# 20242_NLP
# Aspect-Based Sentiment Analysis (ABSA) with BERT and T5

## Project Overview

This project focuses on Aspect-Based Sentiment Analysis (ABSA) using transformer-based models such as BERT and T5. The goal is to identify aspect terms in sentences and classify their corresponding sentiment polarities (positive, negative, neutral, conflict).

We use the **SemEval 2016** dataset, which contains restaurant and laptop reviews annotated with aspect terms and sentiment labels.

## Dataset

- **Source:** SemEval 2016 
- **Language:** English
- **Annotations:**
  - Aspect terms with their positions in sentences.
  - Sentiment polarity labels (positive, negative, neutral) for each aspect term.

## Folder Structure


- The `/model` folder contains the trained models.
- The `/data` folder contains the dataset used for training and evaluation.
- The `/src` folder contains all source code.

## Pre-trained Models

If you want to use pre-trained models directly without training, download them from the following Google Drive link:

[Google Drive - Pretrained ABSA Models](https://drive.google.com/drive/folders/1Cc0ZZx7L9Zw7ozcMC2MYCb86DBGhbwgQ?usp=drive_link)

## Installation

Make sure you have Python 3.8+ installed. Then install required packages using:


1. Clone the repository:
   ```bash
   git clone https://github.com/Uyrjn/20242_NLP.git
   cd 20242_NLP

2.  Then install required packages using:

pip install -r requirements.txt


