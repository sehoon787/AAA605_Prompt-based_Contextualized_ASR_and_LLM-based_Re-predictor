import torch
import torch.nn as nn

class PrunedRNNTLoss(nn.Module):
    def __init__(self, prune_range=5, reduction="mean"):
        super().__init__()
        self.prune_range = prune_range
        self.reduction = reduction

    def forward(self, log_probs, targets, logit_lengths, target_lengths):
        """
        log_probs: (B, T, U, V) -- output from joint network after log_softmax
        targets: (B, U)
        logit_lengths: (B,)
        target_lengths: (B,)
        """

        B, T, U, V = log_probs.size()
        losses = []

        for b in range(B):
            t_len = logit_lengths[b].item()
            u_len = target_lengths[b].item()
            target = targets[b, :u_len]

            # Initialize alpha (forward variable)
            alpha = torch.full((t_len+1, u_len+1), float('-inf'), device=log_probs.device)
            alpha[0, 0] = 0.0

            # log-domain에서 재귀적으로 logaddexp로 forward sum 수행
            for t in range(t_len+1):
                u_range = range(max(0, t - self.prune_range), min(u_len+1, t + self.prune_range + 1))
                for u in u_range:
                    if t > 0:
                        prev_alpha = alpha[t-1, u]
                        p_blank = log_probs[b, t-1, u, 0]  # blank index assumed to be 0
                        alpha[t, u] = torch.logaddexp(alpha[t, u], prev_alpha + p_blank)

                    if u > 0:
                        prev_alpha = alpha[t, u-1]
                        label_id = target[u-1].item()
                        p_label = log_probs[b, t, u-1, label_id]
                        alpha[t, u] = torch.logaddexp(alpha[t, u], prev_alpha + p_label)

            losses.append(-alpha[t_len, u_len])

        loss = torch.stack(losses)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
