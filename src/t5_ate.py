import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.optim import AdamW
from tqdm import tqdm
df = pd.read_csv("restaurants_train_sb1 (1).csv")
df = df.dropna(subset=["sentence_text", "target"]).copy()
df["input"] = "extract aspects: " + df["sentence_text"]
df["output"] = df["target"] 
class T5ATEDataset(Dataset):
    def __init__(self, inputs, targets, tokenizer, max_len=128):
        self.inputs = inputs
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_enc = self.tokenizer(
            self.inputs[idx], padding="max_length", truncation=True,
            max_length=self.max_len, return_tensors="pt"
        )
        target_enc = self.tokenizer(
            self.targets[idx], padding="max_length", truncation=True,
            max_length=64, return_tensors="pt"
        )
        labels = target_enc["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_enc["input_ids"].squeeze(),
            "attention_mask": input_enc["attention_mask"].squeeze(),
            "labels": labels
        }
tokenizer = T5Tokenizer.from_pretrained("t5-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
train_ds = T5ATEDataset(train_df["input"].tolist(), train_df["output"].tolist(), tokenizer)
val_ds = T5ATEDataset(val_df["input"].tolist(), val_df["output"].tolist(), tokenizer)
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=4)
model = T5ForConditionalGeneration.from_pretrained("t5-base").to(device)
