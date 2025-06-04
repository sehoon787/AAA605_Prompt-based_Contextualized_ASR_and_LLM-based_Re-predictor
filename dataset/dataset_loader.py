import os
import tarfile
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
        dataset_split = "test.clean"
    """

    SPLIT_MAP = {
        "train.clean.100": "train-clean-100",
        "validation.clean": "dev-clean",
        "test.clean": "test-clean",
    }

    def __init__(self, tokenizer, dataset_split="train.clean.100", max_prompt_len=32):
        self.dataset_split = dataset_split
        self.tokenizer = tokenizer
        self.max_prompt_len = max_prompt_len

        # 데이터 경로
        self.base_data_dir = r"C:\Users\Administrator\Desktop\ku\1-2\AAA605_Prompt-based_Contextualized_ASR_and_LLM-based_Re-predictor\data"
        self.extract_dir = os.path.join(self.base_data_dir, "LibriSpeech")
        self.tar_filename = self.SPLIT_MAP[self.dataset_split] + ".tar.gz"

        self._extract_if_needed()

        # 압축 해제된 로컬 데이터셋 로드
        self.dataset = load_dataset(
            "librispeech_asr",
            split=self.dataset_split,
            data_dir=self.extract_dir,
            trust_remote_code=True
        )

        self.dataset = self.dataset.filter(lambda x: x['audio'] is not None and x['text'] is not None)

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_mels=80
        )

    def _extract_if_needed(self):
        tar_path = os.path.join(self.base_data_dir, self.tar_filename)
        target_dir = os.path.join(self.extract_dir, self.SPLIT_MAP[self.dataset_split])

        if not os.path.exists(target_dir):
            print(f"Extracting {self.tar_filename}...")
            os.makedirs(self.extract_dir, exist_ok=True)
            with tarfile.open(tar_path, "r:gz") as tar:
                safe_extract(tar, path=self.base_data_dir)
            print(f"Extracted {self.SPLIT_MAP[self.dataset_split]} complete.")
        else:
            print(f"{self.SPLIT_MAP[self.dataset_split]} already extracted.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        speech_array, sr = torchaudio.load(sample['audio']['path'])
        speech_array = speech_array.mean(0)
        speech_tensor = self.mel_transform(speech_array)

        prompt_text = "You are recognizing the following utterance."
        sentence_text = sample["text"]

        prompt_encoding = self.tokenizer(prompt_text, return_tensors="pt", padding="max_length",
                                         max_length=self.max_prompt_len, truncation=True)
        label_encoding = self.tokenizer(sentence_text, return_tensors="pt")

        return (
            speech_tensor.transpose(0, 1),
            prompt_encoding["input_ids"].squeeze(0),
            prompt_encoding["attention_mask"].squeeze(0),
            label_encoding["input_ids"].squeeze(0)[1:-1]
        )

import pathlib

def safe_extract(tar, path="."):
    for member in tar.getmembers():
        member_path = os.path.join(path, member.name)
        abs_path = pathlib.Path(member_path).resolve()
        if not str(abs_path).startswith(str(pathlib.Path(path).resolve())):
            raise Exception("Attempted Path Traversal in Tar File")
    tar.extractall(path=path)

def collate_fn(batch):
    speech, input_ids, attention_mask, labels = zip(*batch)

    speech = torch.nn.utils.rnn.pad_sequence(speech, batch_first=True)
    input_ids = torch.stack(input_ids)
    attention_mask = torch.stack(attention_mask)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)

    input_lengths = torch.tensor([s.size(0) for s in speech])
    label_lengths = torch.tensor([l.size(0) for l in labels])

    return speech, input_ids, attention_mask, labels, input_lengths, label_lengths
