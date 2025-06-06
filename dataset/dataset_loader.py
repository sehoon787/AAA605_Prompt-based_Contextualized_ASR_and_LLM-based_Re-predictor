import os
import glob
import torch
import torchaudio
from torch.utils.data import Dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers, decoders
from config.asr_config import config
from dataset.bpe_tokenizer import BPEAutoTokenizer
import random

class ASRDataset(Dataset):
    def __init__(self, dataset_split="train-clean-100", max_prompt_len=32, train_tokenizer: bool=False):
        self.max_prompt_len = max_prompt_len

        self.base_data_dir = r"D:\ku\1-2\AAA605_Prompt-based_Contextualized_ASR_and_LLM-based_Re-predictor\data\LibriSpeech"
        self.data_dir = os.path.join(self.base_data_dir, dataset_split)

        self.transcripts = self._load_transcripts()
        self.samples = list(self.transcripts.items())

        self.tokenizer_path = os.path.join(self.base_data_dir, "bpe_tokenizer", "tokenizer.json")
        if train_tokenizer:
            self._prepare_tokenizer()
        self.tokenizer = BPEAutoTokenizer(self.tokenizer_path, max_length=32)

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_mels=config['speech_input_dim']
        )

    def _load_transcripts(self):
        transcript_dict = {}
        transcript_files = glob.glob(os.path.join(self.data_dir, "*", "*", "*.trans.txt"))

        for trans_file in transcript_files:
            with open(trans_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(" ", 1)
                    if len(parts) == 2:
                        utt_id, text = parts

                        # prior_prompt는 기존 text 사용 (실제 모델에선 과거 대화 context로 사용 가능)
                        prior_prompt = text
                        # instruct_prompt는 사용자 주입 (여기서는 예시로 random 생성)
                        instruct_prompt = self._generate_dummy_instruct_prompt()

                        transcript_dict[utt_id] = (text, prior_prompt, instruct_prompt)

        if len(transcript_dict) == 0:
            raise ValueError(f"No transcript data found in {self.data_dir}. Please check directory structure.")
        return transcript_dict

    def _generate_dummy_instruct_prompt(self):
        # 임의의 instruct prompt 샘플 생성
        candidates = [
            "This conversation is about science.",
            "The speaker is discussing literature.",
            "This is a story about a family.",
            "The topic is historical events.",
            "The following sentence is an audiobook narration."
        ]
        return random.choice(candidates)

    def _prepare_tokenizer(self):
        tokenizer_dir = os.path.join(self.base_data_dir, "bpe_tokenizer")
        os.makedirs(tokenizer_dir, exist_ok=True)

        all_transcripts = [v[0] for v in self.transcripts.values()]

        temp_corpus = os.path.join(self.base_data_dir, "bpe_corpus.txt")
        with open(temp_corpus, "w", encoding="utf-8") as f:
            for text in all_transcripts:
                f.write(text + "\n")

        tokenizer = Tokenizer(models.BPE())
        tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC()])
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.decoder = decoders.BPEDecoder()

        trainer = trainers.BpeTrainer(
            vocab_size=1024,
            special_tokens=["<pad>", "<unk>", "<s>", "</s>", "<blank>"]
        )
        tokenizer.train([temp_corpus], trainer=trainer)

        tokenizer.save(self.tokenizer_path)
        print(f"Tokenizer saved to {self.tokenizer_path}")

        os.remove(temp_corpus)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        utt_id, (text, prior_prompt, instruct_prompt) = self.samples[idx]
        speaker_id, chapter_id, utt_num = utt_id.split("-")
        audio_path = os.path.join(self.data_dir, speaker_id, chapter_id, utt_id + ".flac")

        speech_array, sr = torchaudio.load(audio_path)
        speech_array = speech_array.mean(0)
        speech_tensor = self.mel_transform(speech_array)

        # utterance-level prompt = prior + instruct
        utterance_prompt = prior_prompt + " " + instruct_prompt

        utterance_encoding = self.tokenizer(utterance_prompt, return_tensors="pt", padding="max_length",
                                            max_length=self.max_prompt_len, truncation=True)

        label_encoding = self.tokenizer(text, return_tensors="pt")

        return (
            speech_tensor.transpose(0, 1),
            utterance_encoding["input_ids"].squeeze(0),
            utterance_encoding["attention_mask"].squeeze(0),
            label_encoding["input_ids"].squeeze(0)
        )

def collate_fn(batch):
    speech, utterance_input_ids, utterance_attention_mask, labels = zip(*batch)

    speech = torch.nn.utils.rnn.pad_sequence(speech, batch_first=True)
    utterance_input_ids = torch.stack(utterance_input_ids)
    utterance_attention_mask = torch.stack(utterance_attention_mask)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)

    input_lengths = torch.tensor([s.size(0) for s in speech])
    label_lengths = torch.tensor([l.size(0) for l in labels])

    return speech, utterance_input_ids, utterance_attention_mask, labels, input_lengths, label_lengths
