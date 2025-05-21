import torch
from warp_rna import rna_loss
        
class RNALoss(torch.nn.Module):
    def __init__(self, blank=0, reduction="mean"):
        super(RNALoss, self).__init__()
        self.blank=blank
        self.reduction=reduction
    
    def forward(self, logits, targets, fbank_len, text_len): 
        """
        Args: 
            logits: [B, T, U, vocab_size]
            targets: [B, U]
            fbank_len: [B]
            text_len: [B]
        """
        logits = logits.log_softmax(dim=-1)
        
        loss = rna_loss(log_probs= logits, 
                        labels= targets[:, :].int(), 
                        frames_lengths= fbank_len.int(), 
                        labels_lengths= text_len.int(), 
                        blank=self.blank
                        )

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

