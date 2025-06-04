import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config.asr_config import config
from dataset.dataset_loader import ASRDataset, collate_fn
from models.asr_model import ASRModel
from models.losses.pruned_rnnt_loss import PrunedRNNTLoss
from utils.hf_auth import huggingface_login

# 필요시
# huggingface_login()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 준비
tokenizer = AutoTokenizer.from_pretrained(config["pretrained_model_name"])
train_dataset = ASRDataset(tokenizer, dataset_split="train-clean-100")
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

# 모델 및 optimizer 준비
model = ASRModel(config).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
pruned_loss_fn = PrunedRNNTLoss(reduction="mean", prune_range=5).to(device)

# 체크포인트 디렉토리 생성
os.makedirs("checkpoints", exist_ok=True)

# 학습 루프
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

        # forward
        logits = model(speech_input, input_ids, attention_mask, label_tokens)
        log_probs = nn.functional.log_softmax(logits, dim=-1)

        # Custom Pruned RNNT loss function
        loss = pruned_loss_fn(
            log_probs=log_probs,
            targets=label_tokens,
            logit_lengths=input_lengths,
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
        'losses': avg_loss,
        'config': config
    }, checkpoint_path)

    print(f"Checkpoint saved at: {checkpoint_path}")
