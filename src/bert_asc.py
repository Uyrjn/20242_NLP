class AspectSentimentDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label2id = {'positive': 0,'negative': 1,'neutral': 2,'conflict': 3}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        sentence = item['Sentence']
        aspect = item['Aspect Term']
        sentiment = item['polarity']
        
        inputs = self.tokenizer(
            sentence,
            aspect,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        
        label = self.label2id[sentiment]
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }
