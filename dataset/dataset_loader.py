import os
import torch
import torchaudio
from datasets import load_dataset, DownloadConfig
import aiohttp

class ASRDataset(torch.utils.data.Dataset):
    """
    학습시:
    dataset_split = "train.clean.100"
    검증시:
    dataset_split = "validation.clean"
    실험시:
    dataset_split = "test.clean"  # 또는 "test.other"
    """
    def __init__(self, tokenizer, dataset_split="train.clean.100", max_prompt_len=32):
        # 캐시 경로 설정 (변경 가능)
        cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
        download_config = DownloadConfig(
            resume_download=True,
            max_retries=10  # 실패 시 10번 재시도
        )

        self.dataset = load_dataset(
            "librispeech_asr",
            split=dataset_split,
            cache_dir=cache_dir,
            download_config=download_config,
            storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600)}}
        )
        self.tokenizer = tokenizer
        self.max_prompt_len = max_prompt_len

        self.dataset = self.dataset.filter(lambda x: x['audio'] is not None and x['text'] is not None)

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_mels=80
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        # audio는 librosa에서 array가 아니라 path or tensor임
        speech_array, sr = torchaudio.load(sample['audio']['path'])
        speech_array = speech_array.mean(0)  # mono 변환 (LibriSpeech는 대부분 mono지만 안전을 위해)
        speech_tensor = self.mel_transform(speech_array)

        prompt_text = "You are recognizing the following utterance."
        sentence_text = sample["text"]

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
