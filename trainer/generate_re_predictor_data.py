# @deprecated: NOT USED
import torch
import json
from torch.utils.data import DataLoader

from config.asr_config import config
from dataset.dataset_loader import ASRDataset, collate_fn
from models.asr_model import ASRModel
from models.inference.beam_search_decoder import RNNTBeamSearchDecoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_path = "checkpoints/asr_model_epoch_10.pt"
checkpoint = torch.load(checkpoint_path, map_location=device)
config = checkpoint['config']

model = ASRModel(config).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

train_dataset = ASRDataset(dataset_split="train-clean-100")
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

beam_decoder = RNNTBeamSearchDecoder(
    encoder=model.encoder,
    decoder=model.decoder,
    tokenizer=train_dataset.tokenizer,
    beam_size=5,
    device=device
)

re_predictor_data = []

for batch in train_loader:
    speech_input, input_ids, attention_mask, labels, input_lengths, label_lengths = batch

    speech_input = speech_input.to(device)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)

    gt_text = train_dataset.tokenizer.decode(labels[0].tolist(), skip_special_tokens=True)

    n_best = beam_decoder.recognize(speech_input, input_ids, attention_mask)

    candidates = []
    for hyp in n_best:
        candidates.append({
            'text': hyp['text'],
            'score': hyp['score']
        })

    best_idx = 0
    best_score = -1
    for idx, cand in enumerate(candidates):
        match_score = int(cand['text'].strip().lower() == gt_text.strip().lower())
        if match_score > best_score:
            best_score = match_score
            best_idx = idx

    re_predictor_data.append({
        "utterance_prompt": "You are recognizing the following utterance.",
        "candidates": candidates,
        "label": best_idx
    })

with open("data/re_predictor_training_data.json", "w") as f:
    json.dump(re_predictor_data, f, indent=2)

print("Re-Predictor fine-tuning data generation complete ✅")
