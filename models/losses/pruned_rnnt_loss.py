import torch
import torch.nn as nn

class PrunedRNNTLoss(nn.Module):
    def __init__(self, prune_range=5, reduction="mean"):
        super().__init__()
        self.prune_range = prune_range
        self.reduction = reduction

    def forward(self, log_probs, targets, logit_lengths, target_lengths):
        B, T, U, V = log_probs.size()
        losses = []

        for b in range(B):
            t_len = logit_lengths[b].item()
            u_len = target_lengths[b].item()
            target = targets[b, :u_len]

            alpha = torch.full((t_len+1, u_len+1), float('-inf'), device=log_probs.device)
            alpha[0, 0] = 0.0

            for t in range(t_len+1):
                u_range = range(max(0, t - self.prune_range), min(u_len+1, t + self.prune_range + 1))

                for u in u_range:
                    candidates = []

                    if t > 0 and t-1 < T and u < U:
                        prev_alpha = alpha[t-1, u]
                        p_blank = log_probs[b, t-1, u, 0]
                        candidates.append(prev_alpha + p_blank)

                    if u > 0 and t < T and u-1 < U:
                        prev_alpha = alpha[t, u-1]
                        label_id = target[u-1].item()
                        p_label = log_probs[b, t, u-1, label_id]
                        candidates.append(prev_alpha + p_label)

                    if len(candidates) > 0:
                        stacked = torch.stack(candidates)
                        if torch.all(torch.isneginf(stacked)):
                            # 모든 candidate가 -inf이면 그냥 유지
                            new_alpha = torch.tensor(float('-inf'), device=log_probs.device)
                        else:
                            new_alpha = torch.logsumexp(stacked, dim=0)
                        alpha[t, u] = new_alpha

            losses.append(-alpha[t_len, u_len])

        loss = torch.stack(losses)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
