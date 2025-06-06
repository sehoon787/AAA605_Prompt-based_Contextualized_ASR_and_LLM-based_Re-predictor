import os
import pdb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from jiwer import wer, cer
from transformers import get_linear_schedule_with_warmup

from config.asr_config import config
from dataset.dataset_loader import ASRDataset, collate_fn
from models.asr_model import ASRModel
from models.losses.pruned_rnnt_loss import RNNTLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset 준비
train_dataset = ASRDataset(dataset_split="train-clean-100", train_tokenizer=True)
val_dataset = ASRDataset(dataset_split="dev-clean")

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

# 모델, optimizer, loss 준비
model = ASRModel(config).to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)  # AdamW 추천
rnnt_loss_fn = RNNTLoss().to(device)

# Warmup + Linear Decay 스케줄러 준비
num_epochs = 10
total_steps = len(train_loader) * num_epochs
warmup_steps = int(0.1 * total_steps)  # 10% warmup

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# AMP (mixed precision) 사용
scaler = GradScaler()

# Dummy Inference용 디코더 (Beam Search 디코더 구현 전까지 placeholder)
def simple_decode(log_probs):
    return ["PLACEHOLDER" for _ in range(log_probs.size(0))]

os.makedirs("checkpoints", exist_ok=True)

# Train loop
for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch in progress_bar:
        speech_input, utterance_ids, utterance_mask, label_tokens, input_lengths, label_lengths = batch

        speech_input = speech_input.to(device)
        utterance_ids = utterance_ids.to(device)
        utterance_mask = utterance_mask.to(device)
        label_tokens = label_tokens.to(device)
        input_lengths = input_lengths.to(device)
        label_lengths = label_lengths.to(device)

        optimizer.zero_grad()

        print("speech_input NaN:", torch.isnan(speech_input).any())
        print("speech_input Max:", speech_input.max(), "Min:", speech_input.min())

        logits = model(
            speech_input,
            utterance_ids, utterance_mask,
            label_tokens
        )
        print("Logits NaN Check:", torch.isnan(logits).any())
        print("Logits Inf Check:", torch.isinf(logits).any())
        print("Logits Max:", logits.max().item(), "Min:", logits.min().item())

        log_probs = nn.functional.log_softmax(logits, dim=-1)
        loss = rnnt_loss_fn(
            log_probs=log_probs,
            targets=label_tokens,
            logit_lengths=input_lengths,
            target_lengths=label_lengths
        )

        try:
            scaler.scale(loss).backward()
        except Exception as err:
            pdb.set_trace()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        progress_bar.set_postfix(batch_loss=loss.item())

    avg_train_loss = total_loss / len(train_loader)
    current_lr = scheduler.get_last_lr()[0]

    # Validation
    model.eval()
    val_loss = 0
    hyp_list, ref_list = [], []

    with torch.no_grad():
        for batch in val_loader:
            speech_input, utterance_ids, utterance_mask, label_tokens, input_lengths, label_lengths = batch

            speech_input = speech_input.to(device)
            utterance_ids = utterance_ids.to(device)
            utterance_mask = utterance_mask.to(device)
            label_tokens = label_tokens.to(device)
            input_lengths = input_lengths.to(device)
            label_lengths = label_lengths.to(device)

            logits = model(speech_input, utterance_ids, utterance_mask, label_tokens)
            log_probs = nn.functional.log_softmax(logits, dim=-1)
            loss = rnnt_loss_fn(
                log_probs=log_probs,
                targets=label_tokens,
                logit_lengths=input_lengths,
                target_lengths=label_lengths
            )

            val_loss += loss.item()
            hyps = simple_decode(log_probs)
            refs = train_dataset.tokenizer.batch_decode(label_tokens, skip_special_tokens=True)

            hyp_list.extend(hyps)
            ref_list.extend(refs)

    avg_val_loss = val_loss / len(val_loader)
    wer_score = wer(ref_list, hyp_list)
    cer_score = cer(ref_list, hyp_list)

    print(f"\nEpoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
          f"WER: {wer_score*100:.2f}% | CER: {cer_score*100:.2f}% | LR: {current_lr:.8f}")

    # 체크포인트 저장
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
