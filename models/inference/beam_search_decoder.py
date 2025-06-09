import torch
import torch.nn.functional as F
from copy import deepcopy

class RNNTBeamSearchDecoder:
    def __init__(self, encoder, decoder, tokenizer, blank_id=0, beam_size=5, device="cuda"):
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.blank_id = blank_id
        self.beam_size = beam_size
        self.device = device

    def recognize(self, speech_input, input_ids, attention_mask, max_output_length=100):
        self.encoder.eval()
        self.decoder.eval()

        with torch.no_grad():
            encoder_out = self.encoder(speech_input, input_ids, attention_mask)
            T_enc = encoder_out.size(1)

            beams = [{'tokens': [], 'score': 0.0, 'hidden': None}]

            for t in range(T_enc):
                new_beams = []
                enc_t = encoder_out[:, t:t+1, :]

                for beam in beams:
                    prev_tokens = torch.tensor(
                        beam['tokens'] if beam['tokens'] else [self.blank_id],
                        dtype=torch.long, device=self.device
                    ).unsqueeze(0)

                    predictor_out, hidden = self.decoder.prediction_net(prev_tokens, beam['hidden'])
                    joint_out = self.decoder.joint_net(enc_t, predictor_out[:, -1:, :])
                    log_probs = F.log_softmax(joint_out.squeeze(1).squeeze(1), dim=-1).squeeze(0)

                    topk_log_probs, topk_ids = torch.topk(log_probs, self.beam_size)

                    for i in range(self.beam_size):
                        new_token = topk_ids[i].item()
                        new_score = beam['score'] + topk_log_probs[i].item()
                        new_tokens = deepcopy(beam['tokens'])

                        if new_token == self.blank_id:
                            new_beams.append({'tokens': new_tokens, 'score': new_score, 'hidden': beam['hidden']})
                        else:
                            new_tokens.append(new_token)
                            new_beams.append({'tokens': new_tokens, 'score': new_score, 'hidden': hidden})

                beams = sorted(new_beams, key=lambda x: x['score'], reverse=True)[:self.beam_size]

        n_best = sorted(beams, key=lambda x: x['score'], reverse=True)
        decoded_results = []
        for hyp in n_best:
            text = self.tokenizer.decode(hyp['tokens'])
            decoded_results.append({'text': text, 'score': hyp['score'], 'tokens': hyp['tokens']})

        return decoded_results
