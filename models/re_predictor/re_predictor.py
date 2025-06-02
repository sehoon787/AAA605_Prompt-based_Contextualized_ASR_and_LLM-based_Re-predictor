import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class RePredictor:
    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B-Instruct", device="cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.model.eval()

    def rerank(self, utterance_prompt, word_prompt, candidates, formatter):
        prompt = formatter.format(utterance_prompt, word_prompt, candidates)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=0.1,
                top_p=0.9
            )

        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result
