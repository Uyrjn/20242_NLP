import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
df = pd.read_csv("restaurants_train_merged_with_mixed.csv")
df = df.dropna(subset=["sentence_text", "polarity"]).copy()

df_with_aspect = df.dropna(subset=["target"]).copy()
df_with_aspect["input"] = "aspect: " + df_with_aspect["target"] + " | category: " + df_with_aspect["category"] + " | context: " + df_with_aspect["sentence_text"]

df_no_aspect = df[df["target"].isna()].copy()
df_no_aspect["input"] = "category: " + df_no_aspect["category"] + " | context: " + df_no_aspect["sentence_text"]

sentiment_df = pd.concat([df_with_aspect, df_no_aspect])
sentiment_df["output"] = sentiment_df["polarity"]
class T5SentimentDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.inputs = df["input"].tolist()
        self.targets = df["output"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_encoding = self.tokenizer(
            self.inputs[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        target_encoding = self.tokenizer(
            self.targets[idx],
            max_length=16,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": input_encoding.input_ids.squeeze(),
            "attention_mask": input_encoding.attention_mask.squeeze(),
            "labels": target_encoding.input_ids.squeeze()
        }
