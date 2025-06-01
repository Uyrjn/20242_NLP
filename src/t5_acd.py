import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.model_selection import train_test_split
df = pd.read_csv("restaurants_train_sb1 (1).csv")
df = df.dropna(subset=["sentence_text", "category", "target"]).copy()

# ACD-direct (cả câu)
df_direct = df.groupby("sentence_text")["category"].apply(lambda x: ", ".join(set(x))).reset_index()
df_direct["acd_direct_input"] = "detect category: " + df_direct["sentence_text"]
df_direct["acd_direct_output"] = df_direct["category"]


# ACD-aspect (từng aspect cụ thể)
grouped_df = df.groupby(['sentence_text', 'target'])['category'].apply(lambda x: ', '.join(set(x))).reset_index()
grouped_df["acd_input"] = "aspect: " + grouped_df["target"] + " | context: " + grouped_df["sentence_text"]
grouped_df["acd_output"] = grouped_df["category"]

tokenizer = T5Tokenizer.from_pretrained("t5-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class T5ACD_Dataset(Dataset):
    def __init__(self, inputs, targets, tokenizer, max_len=128):
        self.inputs = inputs
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_enc = self.tokenizer(
            self.inputs[idx], padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")
        target_enc = self.tokenizer(
            self.targets[idx], padding="max_length", truncation=True, max_length=64, return_tensors="pt")

        labels = target_enc["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_enc["input_ids"].squeeze(),
            "attention_mask": input_enc["attention_mask"].squeeze(),
            "labels": labels
        }
