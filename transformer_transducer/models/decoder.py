from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .encoder import TransformerTransducerLayer

class TransformerTransducerDecoder(nn.Module):
    """Implements the Transformer-Transducer decoder with relative truncated
    multi-head self attention as described in https://arxiv.org/abs/2002.02562

    Args:

        vocab_size (int): The vocabulary size.

        n_layers (int): The number of transformer encoder layers with truncated
        self attention and relative positional encoding.

        d_model (int): The model dimensionality.

        ff_size (int): The feed forward inner layer dimensionality.

        h (int): The number of heads in the attention mechanism.

        left_size (int): The size of the left window that each time step is
        allowed to look at.

        right_size (int): The size of the right window that each time step is
        allowed to look at.

        p_dropout (float): The dropout rate.

        masking_value (float, optional): The value to use for masking padded
        elements. Defaults to -1e15.
    """

    def __init__(
        self,
        vocab_size: int,
        n_layers: int,
        d_model: int,
        ff_size: int,
        h: int,
        left_size: int,
        right_size: int,
        p_dropout: float,
        masking_value: int = -1e15,
    ) -> None:
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.enc_layers = nn.ModuleList(
            [
                TransformerTransducerLayer(
                    d_model=d_model,
                    ff_size=ff_size,
                    h=h,
                    left_size=left_size,
                    right_size=right_size,
                    p_dropout=p_dropout,
                    masking_value=masking_value,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        """Passes the input `x` through the decoder layers.

        Args:

            x (Tensor): The input tensor of shape [B, M]

            mask (Tensor): The input boolean mask of shape [B, M], where it's True
            if there is no padding.

        Returns:

            Tuple[Tensor, Tensor]: A tuple where the first element is the encoded text of shape
            [B, M, d_model] and the second element is the lengths of shape [B].
        """
        lengths = mask.sum(dim=-1)
        out = self.emb(x)

        for layer in self.enc_layers:
            out = layer(out, mask)
        return out, lengths