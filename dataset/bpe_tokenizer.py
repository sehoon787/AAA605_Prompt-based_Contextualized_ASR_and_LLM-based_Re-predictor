import torch
from tokenizers import Tokenizer

class BPEAutoTokenizer:
    def __init__(self, tokenizer_path, max_length=32):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_length = max_length

        self.pad_token_id = self.tokenizer.token_to_id("<pad>")
        self.blank_token_id = self.tokenizer.token_to_id("<blank>")
        self.unk_token_id = self.tokenizer.token_to_id("<unk>")

    def __call__(self, text, return_tensors=None, padding=None, max_length=None, truncation=False):
        max_length = max_length or self.max_length

        encoding = self.tokenizer.encode(text)
        ids = encoding.ids[:max_length]

        if padding == "max_length":
            ids += [self.pad_token_id] * (max_length - len(ids))

        attention_mask = [1 if id != self.pad_token_id else 0 for id in ids]

        result = {
            "input_ids": torch.tensor([ids]),
            "attention_mask": torch.tensor([attention_mask])
        }

        return result

    def batch_decode(self, token_ids_batch, skip_special_tokens=True):
        results = []
        for token_ids in token_ids_batch:
            ids = token_ids.cpu().numpy().tolist()
            if skip_special_tokens:
                ids = [id for id in ids if id != self.pad_token_id and id != self.blank_token_id]
            results.append(self.tokenizer.decode(ids))
        return results
