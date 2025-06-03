import torch
import random
import spacy
import nltk
from transformers import AutoTokenizer
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

tokenizer = AutoTokenizer.from_pretrained(config["pretrained_model_name"])

beam_decoder = RNNTBeamSearchDecoder(
    encoder=model.encoder,
    decoder=model.decoder,
    tokenizer=tokenizer,
    beam_size=5,
    device=device
)

formatter = PromptFormatter()
re_predictor = RePredictor(model_name="meta-llama/Meta-Llama-3-8B-Instruct", device=device)

# Dataset 불러오기 (test.clean 그대로 활용)
test_dataset = ASRDataset(tokenizer, dataset_split="test-clean")
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

# Bias list 추출 함수
def extract_bias_list(text, N):
    doc = nlp(text)
    candidates = [ent.text for ent in doc.ents]
    candidates += [token.text for token in doc if token.pos_ == "NOUN"]
    candidates = list(set(candidates))
    random.shuffle(candidates)
    return candidates[:N]

# 실험 파라미터
bias_sizes = [10, 100, 500, 1000]
results = {N: [] for N in bias_sizes}

# Inference + 실험 반복
for batch in test_loader:
    speech_input, input_ids, attention_mask, labels, input_lengths, label_lengths = batch

    speech_input = speech_input.to(device)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    reference = tokenizer.decode(labels[0], skip_special_tokens=True)

    # N-best 후보 생성
    n_best = beam_decoder.recognize(speech_input, input_ids, attention_mask)

    for N in bias_sizes:
        bias_list = extract_bias_list(reference, N)
        utterance_prompt = "You are recognizing the following utterance."

        result = re_predictor.rerank(
            utterance_prompt, bias_list, n_best, formatter
        )

        # LLM 출력 후처리 (LLM output에서 Answer 부분만 추출)
        hypothesis = result.split("Answer:")[-1].strip() if "Answer:" in result else result

        error = wer(reference.lower(), hypothesis.lower())
        results[N].append(error)

    # 현재 샘플 하나만 테스트 (디버깅용), 전체 실험시 break 제거
    break

# 중간 결과 출력
for N in bias_sizes:
    avg_wer = sum(results[N]) / len(results[N])
    print(f"Bias List N={N} | Average WER: {avg_wer*100:.2f}%")
