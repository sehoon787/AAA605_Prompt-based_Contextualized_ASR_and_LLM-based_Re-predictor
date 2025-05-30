import torch
from torch.utils.data import Dataset

class RePredictorDataset(torch.utils.data.Dataset):
    def __init__(self, data, formatter, tokenizer, max_length=512):
        """
        data: list of dict {
            'utterance_prompt': str,
            'candidates': list of dict { 'text': str, 'score': float },
            'label': int (index of correct hypothesis)
        }
        """
        self.data = data
        self.formatter = formatter
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = self.formatter.format(item['utterance_prompt'], item['candidates'])
        encoding = self.tokenizer(prompt, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(item["label"], dtype=torch.long)
        }
