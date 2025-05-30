import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

from models.re_predictor.prompt_formatter import PromptFormatter
from models.re_predictor.re_predictor_dataset import RePredictorDataset
from models.re_predictor.lora_setup import apply_lora

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model = apply_lora(model)

formatter = PromptFormatter()

# Load generated fine-tuning data
with open("data/re_predictor_training_data.json", "r") as f:
    data = json.load(f)

dataset = RePredictorDataset(data, formatter, tokenizer)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
num_training_steps = len(dataloader) * 5
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

model.train()
for epoch in range(5):
    total_loss = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, -1, :]  # Simplified for ranking task

        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} | Loss: {total_loss/len(dataloader):.4f}")
