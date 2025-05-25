class ATE_Dataset(Dataset):
    def __init__(self, df, tokenizer, max_input_len=128, max_output_len=32):
        self.inputs = df['Sentence'].tolist()
        self.targets = df['Aspect Term'].tolist()
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_text = "extract aspects: " + self.inputs[idx]
        target_text = self.targets[idx]

        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_input_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_output_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        labels = target_encoding['input_ids']
        labels[labels == self.tokenizer.pad_token_id] = -100  
        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }
