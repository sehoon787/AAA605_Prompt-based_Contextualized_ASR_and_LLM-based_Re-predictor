import os
import torch
import torchaudio
from torch.utils.data import Dataset
from glob import glob

class ASRDataset(Dataset):
    """
    학습시:
        dataset_split = "train-clean-100"
    검증시:
        dataset_split = "dev-clean"
    실험시:
        dataset_split = "test-clean"  # 또는 "test-other"
    """

    def __init__(self, tokenizer, dataset_split="train-clean-100", max_prompt_len=32):
        self.tokenizer = tokenizer
        self.max_prompt_len = max_prompt_len

        # 경로 지정 (여기에 본인의 데이터 폴더 지정)
        self.base_data_dir = r"C:\Users\Administrator\Desktop\ku\1-2\AAA605_Prompt-based_Contextualized_ASR_and_LLM-based_Re-predictor\data\LibriSpeech"
        self.data_dir = os.path.join(self.base_data_dir, dataset_split)

        # transcript 파일 읽기
        self.transcripts = self._load_transcripts()

        # 파일 리스트 생성
        self.samples = list(self.transcripts.items())

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_mels=80
        )

    def _load_transcripts(self):
        transcript_dict = {}
        transcript_files = glob(os.path.join(self.data_dir, "*", "*", "*.trans.txt"))

        for trans_file in transcript_files:
            with open(trans_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(" ", 1)
                    if len(parts) == 2:
                        utt_id, text = parts
                        transcript_dict[utt_id] = text

        return transcript_dict

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        utt_id, text = self.samples[idx]

        # 오디오 파일 경로
        speaker_id, chapter_id, utt_num = utt_id.split("-")
        audio_path = os.path.join(self.data_dir, speaker_id, chapter_id, utt_id + ".flac")

        # 오디오 로드
        speech_array, sr = torchaudio.load(audio_path)
        speech_array = speech_array.mean(0)  # mono 변환
        speech_tensor = self.mel_transform(speech_array)

        # 프롬프트와 라벨 인코딩
        prompt_text = "You are recognizing the following utterance."
        prompt_encoding = self.tokenizer(prompt_text, return_tensors="pt", padding="max_length",
                                         max_length=self.max_prompt_len, truncation=True)
        label_encoding = self.tokenizer(text, return_tensors="pt")

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
