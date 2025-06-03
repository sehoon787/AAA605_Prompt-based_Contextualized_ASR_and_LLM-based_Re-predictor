import os
import subprocess
import torch
import torchaudio
from datasets import load_dataset

class ASRDataset(torch.utils.data.Dataset):
    """
    학습시:
    dataset_split = "train.clean.100"
    검증시:
    dataset_split = "validation.clean"
    실험시:
    dataset_split = "test.clean"  # 또는 "test.other"
    """

    # huggingface split 명칭 → huggingface-cli include 인자 변환용 매핑
    HF_CLI_SPLIT_MAP = {
        "train.clean.100": "train.clean.100",
        "train.clean.360": "train.clean.360",
        "train.other.500": "train.other.500",
        "validation.clean": "dev.clean",
        "validation.other": "dev.other",
        "test.clean": "test.clean",
        "test.other": "test.other"
    }

    def __init__(self, tokenizer, dataset_split="train.clean.100", max_prompt_len=32):
        # 본인 local 경로
        self.local_data_dir = "C:/Users/Administrator/hf_datasets/librispeech_asr"

        # 사전 다운로드 수행
        self.download_if_needed(dataset_split)

        # 사전 다운로드된 데이터셋 불러오기 (offline)
        self.dataset = load_dataset(
            "librispeech_asr",
            split=dataset_split,
            data_dir=self.local_data_dir,
            trust_remote_code=True
        )

        self.tokenizer = tokenizer
        self.max_prompt_len = max_prompt_len

        self.dataset = self.dataset.filter(lambda x: x['audio'] is not None and x['text'] is not None)

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_mels=80
        )

    def download_if_needed(self, dataset_split):
        # huggingface-cli 명령어 구성
        hf_cli_split = self.HF_CLI_SPLIT_MAP[dataset_split]

        # 사전 다운로드할 경로가 없으면 수행
        if not os.path.exists(self.local_data_dir):
            os.makedirs(self.local_data_dir, exist_ok=True)

        # huggingface-cli download 수행
        print(f"Downloading {dataset_split}... (huggingface-cli download)")
        cmd = [
            "huggingface-cli",
            "download",
            "librispeech_asr",
            "--repo-type", "dataset",
            "--include", hf_cli_split,
            "--local-dir", self.local_data_dir,
            "--local-dir-use-symlinks", "False"
        ]
        subprocess.run(cmd, check=True)
        print(f"Downloaded {dataset_split} complete.")

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
