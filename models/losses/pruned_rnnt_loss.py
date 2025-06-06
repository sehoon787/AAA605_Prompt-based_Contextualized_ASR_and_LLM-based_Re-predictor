import torch
import torchaudio
import torch.nn as nn
# from icefall.loss import PrunedTransducerLoss
#
# class PrunedRNNTLoss(torch.nn.Module):
#     def __init__(self, reduction="mean", prune_range=5):
#         super().__init__()
#         self.loss_fn = PrunedTransducerLoss(
#             reduction=reduction,
#             prune_range=prune_range
#         )
#
#     def forward(self, log_probs, targets, logit_lengths, target_lengths):
#         """
#         log_probs: (B, T, U, V) - log softmax 되어야 함.
#         targets: (B, U)
#         logit_lengths: (B,)
#         target_lengths: (B,)
#         """
#         return self.loss_fn(
#             log_probs=log_probs,
#             logit_lengths=logit_lengths,
#             targets=targets,
#             target_lengths=target_lengths
#         )

class RNNTLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.loss_fn = torchaudio.transforms.RNNTLoss(reduction=reduction)

    def forward(self, log_probs, targets, logit_lengths, target_lengths):
        logit_lengths = torch.full_like(logit_lengths, log_probs.shape[1])

        print("log_probs:", log_probs.shape)
        print("targets:", targets.shape)  # -> (B, U)
        print("logit_lengths:", logit_lengths)  # -> (B,) and int32
        print("target_lengths:", target_lengths)  # -> (B,) and int32

        targets = targets.to(torch.int32)
        logit_lengths = logit_lengths.to(torch.int32)
        target_lengths = target_lengths.to(torch.int32)
        return self.loss_fn(log_probs, targets, logit_lengths, target_lengths)
