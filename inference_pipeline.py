import torch
from torch.utils.data import DataLoader
from dataset.dataset_loader import ASRDataset, collate_fn
from models.asr_model import ASRModel
from models.inference.beam_search_decoder import RNNTBeamSearchDecoder
from models.re_predictor.prompt_formatter import PromptFormatter
from models.re_predictor.re_predictor import RePredictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ASR model
checkpoint_path = "trainer/checkpoints/asr_model_epoch_10.pt"
checkpoint = torch.load(checkpoint_path, map_location=device)
config = checkpoint['config']

model = ASRModel(config).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load test data
test_dataset = ASRDataset(dataset_split="test-clean")
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

beam_decoder = RNNTBeamSearchDecoder(
    encoder=model.encoder,
    decoder=model.decoder,
    tokenizer=test_dataset.tokenizer,
    beam_size=5,
    device=device
)

formatter = PromptFormatter()
re_predictor = RePredictor(model_name="meta-llama/Meta-Llama-3-8B-Instruct", device=device)

for batch in test_loader:
    speech_input, input_ids, attention_mask, labels, input_lengths, label_lengths = batch

    speech_input = speech_input.to(device)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    n_best = beam_decoder.recognize(speech_input, input_ids, attention_mask)

    utterance_prompt = "You are recognizing the following utterance."
    word_prompt = ['some', 'word', 'here']
    result = re_predictor.rerank(
        utterance_prompt,
        word_prompt,
        n_best,
        formatter
    )

    print("----- Inference Result -----")
    for idx, hyp in enumerate(n_best):
        print(f"{idx+1}: {hyp['text']} (score={hyp['score']:.2f})")
    print("Re-Predictor Output:", result)
    break  # 한 샘플만 추론
