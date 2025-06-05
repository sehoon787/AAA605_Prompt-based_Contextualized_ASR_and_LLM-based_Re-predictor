import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
from jiwer import wer, cer

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
val_dataset = ASRDataset(tokenizer, dataset_split="dev-clean")  # validation 추가

# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
# val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
# Only in Windows
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=0)

# 모델 및 optimizer 준비
model = ASRModel(config).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
pruned_loss_fn = PrunedRNNTLoss(reduction="mean", prune_range=5).to(device)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

os.makedirs("checkpoints", exist_ok=True)

# Inference용 간단한 Greedy Decoder (Placeholder)
def simple_decode(log_probs):
    return ["PLACEHOLDER" for _ in range(log_probs.size(0))]

# Train Loop
for epoch in range(1, 11):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch in progress_bar:
        speech_input, input_ids, attention_mask, label_tokens, input_lengths, label_lengths = batch

        speech_input = speech_input.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        label_tokens = label_tokens.to(device)
        input_lengths = input_lengths.to(device)
        label_lengths = label_lengths.to(device)

        optimizer.zero_grad()

        logits = model(speech_input, input_ids, attention_mask, label_tokens)
        log_probs = nn.functional.log_softmax(logits, dim=-1)

        loss = pruned_loss_fn(
            log_probs=log_probs,
            targets=label_tokens,
            logit_lengths=input_lengths,
            target_lengths=label_lengths
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(batch_loss=loss.item())

    avg_train_loss = total_loss / len(train_loader)
    current_lr = optimizer.param_groups[0]['lr']

    # Validation 시작
    model.eval()
    val_loss = 0
    hyp_list, ref_list = [], []

    with torch.no_grad():
        for batch in val_loader:
            speech_input, input_ids, attention_mask, label_tokens, input_lengths, label_lengths = batch

            speech_input = speech_input.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            label_tokens = label_tokens.to(device)
            input_lengths = input_lengths.to(device)
            label_lengths = label_lengths.to(device)

            logits = model(speech_input, input_ids, attention_mask, label_tokens)
            log_probs = nn.functional.log_softmax(logits, dim=-1)

            loss = pruned_loss_fn(
                log_probs=log_probs,
                targets=label_tokens,
                logit_lengths=input_lengths,
                target_lengths=label_lengths
            )
            val_loss += loss.item()

            # 추후 디코더 연결시 decoding 추가 (지금은 placeholder)
            hyps = simple_decode(log_probs)
            refs = tokenizer.batch_decode(label_tokens, skip_special_tokens=True)

            hyp_list.extend(hyps)
            ref_list.extend(refs)

    avg_val_loss = val_loss / len(val_loader)

    # (지금은 dummy WER/CER, 추후 디코더 연동하면 정확 계산 가능)
    wer_score = wer(ref_list, hyp_list)
    cer_score = cer(ref_list, hyp_list)

    print(f"\nEpoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
          f"WER: {wer_score*100:.2f}% | CER: {cer_score*100:.2f}% | LR: {current_lr:.6f}")

    scheduler.step()

    checkpoint_path = f"checkpoints/asr_model_epoch_{epoch}.pt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
        'config': config
    }, checkpoint_path)

    print(f"Checkpoint saved at: {checkpoint_path}\n")
