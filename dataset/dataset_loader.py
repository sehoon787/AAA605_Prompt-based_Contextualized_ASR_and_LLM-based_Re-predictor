# dataset/dataset_loader.py

import torch
import torchaudio
from datasets import load_dataset
from transformers import AutoTokenizer

class ASRDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, dataset_split="trainer", language="en", max_prompt_len=32):
        self.dataset = load_dataset("mozilla-foundation/common_voice_13_0", language, split=dataset_split)
        self.tokenizer = tokenizer
        self.max_prompt_len = max_prompt_len

        self.dataset = self.dataset.filter(lambda x: x['audio'] is not None and x['sentence'] is not None)
        self.resampler = torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        speech_array = sample["audio"]["array"]
        speech_tensor = torch.tensor(speech_array, dtype=torch.float32).unsqueeze(0)
        speech_tensor = self.resampler(speech_tensor).squeeze(0)
        speech_tensor = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80)(speech_tensor)

        prompt_text = "You are recognizing the following utterance."
        sentence_text = sample["sentence"]

        prompt_encoding = self.tokenizer(prompt_text, return_tensors="pt", padding="max_length",
                                         max_length=self.max_prompt_len, truncation=True)
        label_encoding = self.tokenizer(sentence_text, return_tensors="pt")

        return (
            speech_tensor.transpose(0, 1),  # (T, 80)
            prompt_encoding["input_ids"].squeeze(0),
            prompt_encoding["attention_mask"].squeeze(0),
            label_encoding["input_ids"].squeeze(0)[1:-1]
        )

def collate_fn(batch):
    speech, input_ids, attention_mask, labels = zip(*batch)

    speech = torch.nn.utils.rnn.pad_sequence(speech, batch_first=True)
    input_ids = torch.stack(input_ids)
    attention_mask = torch.stack(attention_mask)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)

    input_lengths = torch.tensor([s.size(0) for s in speech])
    label_lengths = torch.tensor([l.size(0) for l in labels])

    return speech, input_ids, attention_mask, labels, input_lengths, label_lengths
