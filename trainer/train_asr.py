import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from icefall.loss import PrunedTransducerLoss

from config.asr_config import config
from dataset.dataset_loader import ASRDataset, collate_fn
from models.asr_model import ASRModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(config["pretrained_model_name"])
train_dataset = ASRDataset(tokenizer, dataset_split="trainer", language="en")
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

model = ASRModel(config).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
pruned_loss_fn = PrunedTransducerLoss(reduction="mean", prune_range=5).to(device)

os.makedirs("checkpoints", exist_ok=True)

for epoch in range(10):
    model.train()
    total_loss = 0

    for batch in train_loader:
        speech_input, input_ids, attention_mask, label_tokens, input_lengths, label_lengths = batch

        speech_input = speech_input.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        label_tokens = label_tokens.to(device)
        input_lengths = input_lengths.to(device)
        label_lengths = label_lengths.to(device)

        optimizer.zero_grad()

        logits = model(speech_input, input_ids, attention_mask, label_tokens)
        logits = logits.permute(0, 1, 2, 3).contiguous()
        log_probs = nn.functional.log_softmax(logits, dim=-1)

        loss = pruned_loss_fn(
            log_probs=log_probs,
            targets=label_tokens,
            input_lengths=input_lengths,
            target_lengths=label_lengths
        )

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")

    checkpoint_path = f"checkpoints/asr_model_epoch_{epoch+1}.pt"
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'config': config
    }, checkpoint_path)

    print(f"Checkpoint saved at: {checkpoint_path}")
