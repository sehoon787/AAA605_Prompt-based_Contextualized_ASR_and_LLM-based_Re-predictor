import torch
import random
import spacy
import nltk
from jiwer import wer
from torch.utils.data import DataLoader
from dataset.dataset_loader import ASRDataset, collate_fn
from models.asr_model import ASRModel
from models.inference.beam_search_decoder import RNNTBeamSearchDecoder
from models.re_predictor.prompt_formatter import PromptFormatter
from models.re_predictor.re_predictor import RePredictor

# 환경 준비
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nlp = spacy.load("en_core_web_sm")
nltk.download('punkt')

# 모델 및 모듈 로드
checkpoint_path = "checkpoints/asr_model_epoch_10.pt"
checkpoint = torch.load(checkpoint_path, map_location=device)
config = checkpoint['config']

model = ASRModel(config).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file('bpe_tokenizer/tokenizer.json')

beam_decoder = RNNTBeamSearchDecoder(
    encoder=model.encoder,
    decoder=model.decoder,
    tokenizer=tokenizer,
    beam_size=6,
    device=device
)

formatter = PromptFormatter()
re_predictor = RePredictor(model_name="meta-llama/Meta-Llama-3-8B-Instruct", device=device)

# Dataset 불러오기 (test-clean 및 test-other 분리 실험)
datasets = {
    "test-clean": ASRDataset(dataset_split="test-clean"),
    "test-other": ASRDataset(dataset_split="test-other")
}

bias_sizes = [10, 100, 500, 1000]
final_results = {}

for dataset_name, dataset in datasets.items():
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    results = {N: [] for N in bias_sizes}

    for batch in test_loader:
        speech_input, input_ids, attention_mask, labels, input_lengths, label_lengths = batch

        speech_input = speech_input.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        reference = tokenizer.decode(labels[0], skip_special_tokens=True)

        n_best = beam_decoder.recognize(speech_input, input_ids, attention_mask)

        for N in bias_sizes:
            def extract_bias_list(text, N):
                doc = nlp(text)
                candidates = [ent.text for ent in doc.ents]
                candidates += [token.text for token in doc if token.pos_ == "NOUN"]
                candidates = list(set(candidates))
                random.shuffle(candidates)
                return candidates[:N]

            bias_list = extract_bias_list(reference, N)
            utterance_prompt = "You are recognizing the following utterance."

            result = re_predictor.rerank(
                utterance_prompt, bias_list, n_best, formatter
            )

            hypothesis = result.split("Answer:")[-1].strip() if "Answer:" in result else result
            error = wer(reference.lower(), hypothesis.lower())
            results[N].append(error)

    final_results[dataset_name] = {}
    for N in bias_sizes:
        final_results[dataset_name][N] = sum(results[N]) / len(results[N])

# 최종 결과 출력 (Baseline 제거 버전)
print(f"{'Test set':<15}{'N=10':<10}{'N=100':<10}{'N=500':<10}{'N=1000':<10}")
for dataset_name in datasets.keys():
    row = f"{dataset_name:<15}"
    for N in bias_sizes:
        row += f"{final_results[dataset_name][N]*100:.2f}%   "
    print(row)
