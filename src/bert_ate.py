class AspectTermDataset(Dataset):
    def __init__(self, sentences, aspect_terms, tokenizer):
        self.sentences = sentences
        self.aspect_terms = aspect_terms
        self.tokenizer = tokenizer
        self.tag2id = {'O':0, 'B-ASP':1, 'I-ASP':2}

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        aspects = self.aspect_terms[idx]
        words = sentence.split()

        tags = ['O'] * len(words)
        for asp in aspects:
            asp_words = asp.split()
            for i in range(len(words) - len(asp_words) + 1):
                if words[i:i+len(asp_words)] == asp_words:
                    tags[i] = 'B-ASP'
                    for j in range(1, len(asp_words)):
                        tags[i+j] = 'I-ASP'
        encoding = self.tokenizer(sentence.split(),
                                  is_split_into_words=True,
                                  return_offsets_mapping=True,
                                  padding='max_length',
                                  truncation=True,
                                  max_length=128,
                                  return_tensors="pt")

        labels = []
        word_ids = encoding.word_ids(batch_index=0)  
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                labels.append(-100)  
            elif word_idx != previous_word_idx:
                labels.append(self.tag2id[tags[word_idx]])
            else:
                labels.append(self.tag2id[tags[word_idx]] if tags[word_idx].startswith('I') else -100)
            previous_word_idx = word_idx

        item = {key: val.squeeze() for key, val in encoding.items()}
        item['labels'] = torch.tensor(labels)
        return item
