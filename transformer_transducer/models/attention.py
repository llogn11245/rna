from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .utils import truncate_attention_mask, add_pos_enc

class MultiHeadAtt(nn.Module):
    """A module that implements the multi-head attention mechanism described in
    https://arxiv.org/abs/1706.03762.

    Args:
        d_model (int): The dimensionality of the model.

        h (int): The number of heads to use in the attention mechanism.

        masking_value (float, optional): The value used for masking. Defaults
        to -1e15.
    """

    def __init__(self, d_model: int, h: int, masking_value: int = -1e15) -> None:
        super().__init__()
        self.h = h
        self.dk = d_model // h
        self.d_model = d_model
        self.masking_value = masking_value
        assert d_model % h == 0, ValueError
        self.query_fc = nn.Linear(in_features=d_model, out_features=d_model)
        self.key_fc = nn.Linear(in_features=d_model, out_features=d_model)
        self.value_fc = nn.Linear(in_features=d_model, out_features=d_model)
        self.softmax = nn.Softmax(dim=-1)

    def _reshape(self, x: Tensor) -> List[Tensor]:
        batch_size, max_len, _ = x.shape
        x = x.view(batch_size, max_len, self.h, self.dk)
        return x

    def _mask(self, att: Tensor, key_mask: Tensor, query_mask: Tensor) -> Tensor:
        key_max_len = key_mask.shape[-1]
        query_max_len = query_mask.shape[-1]
        key_mask = key_mask.repeat(1, query_max_len)
        key_mask = key_mask.view(-1, query_max_len, key_max_len)
        if query_mask.dim() != key_mask.dim():
            query_mask = query_mask.unsqueeze(dim=-1)
        mask = key_mask & query_mask
        mask = mask.unsqueeze(dim=1)
        return att.masked_fill(~mask, self.masking_value)

    def perform_attention(
        self,
        key: Tensor,
        query: Tensor,
        value: Tensor,
        key_mask: Optional[Tensor] = None,
        query_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Performs multi-head attention by computing a weighted sum of the
        values using queries and keys. The weights are computed as a softmax
        over the dot products of the queries and keys for each attention head.
        Optionally, attention can be masked using key and query masks.

        Args:
            key (Tensor): The key input tensor of shape [B, M, d]

            query (Tensor): The query of shape [B, M, d]

            value (Tensor): Teh value tensor of shape [B, M, d]

            key_mask (Tensor, optional): A boolean tensor of shape
            [B, M] where True indicates that the corresponding key position
            contains data, not padding, and should not be masked

            query_mask (Tensor, optional): A boolean tensor of shape
            [B, M] where True indicates that the corresponding query position
            contains data, not padding, and should not be masked

        Returns:
            Tensor: The tensor of shape [B, M, d] resulting from the multi-head
            attention computation.
        """
        key = self._reshape(key)  # B, M, h, dk
        query = self._reshape(query)  # B, M, h, dk
        value = self._reshape(value)  # B, M, h, dk
        key = key.permute(0, 2, 3, 1)  # B, h, dk, M
        query = query.permute(0, 2, 1, 3)  # B, h, M, dk
        value = value.permute(0, 2, 1, 3)  # B, h, M, dk
        att = torch.matmul(query, key)
        if key_mask is not None and query_mask is not None:
            att = self._mask(att=att, key_mask=key_mask, query_mask=query_mask)
        att = self.softmax(att / self.d_model)
        out = torch.matmul(att, value)
        out = out.permute(0, 2, 1, 3)
        out = out.contiguous()
        out = out.view(out.shape[0], out.shape[1], -1)
        return out

    def forward(
        self,
        key: Tensor,
        query: Tensor,
        value: Tensor,
        key_mask: Union[Tensor, None] = None,
        query_mask: Union[Tensor, None] = None,
    ) -> Tensor:
        """passes the input to the multi-head attention by computing a weighted
        sum of the values using queries and keys. The weights are computed as a softmax
        over the dot products of the queries and keys for each attention head.
        Optionally, attention can be masked using key and query masks.

        Args:
            key (Tensor): The key input tensor of shape [B, M, d]

            query (Tensor): The query of shape [B, M, d]

            value (Tensor): Teh value tensor of shape [B, M, d]

            key_mask (Tensor, optional): A boolean tensor of shape
            [B, M] where True indicates that the corresponding key position
            contains data, not padding, and should not be masked

            query_mask (Tensor, optional): A boolean tensor of shape
            [B, M] where True indicates that the corresponding query position
            contains data, not padding, and should not be masked

        Returns:
            Tensor: The tensor of shape [B, M, d] resulting from the multi-head
            attention computation.
        """
        key   = self.key_fc(key)
        query = self.query_fc(query)
        value = self.value_fc(value)

        return self.perform_attention(
            key=key, query=query, value=value, key_mask=key_mask, query_mask=query_mask
        )


class TruncatedSelfAttention(MultiHeadAtt):
    """Builds the truncated self attention module used
    in https://arxiv.org/abs/1910.12977

    Args:

        d_model (int): The model dimension.

        h (int): The number of attention heads.

        left_size (int): The size of the left window that each time step is
        allowed to look at.

        right_size (int): The size of the right window that each time step is
        allowed to look at.

        masking_value (float): The attention masking value.
    """

    def __init__(
        self,
        d_model: int,
        h: int,
        left_size: int,
        right_size: int,
        masking_value: float = -1e15,
    ) -> None:
        super().__init__(d_model=d_model, h=h, masking_value=masking_value)
        self.left_size = left_size
        self.right_size = right_size

    def get_looking_ahead_mask(self, mask: Tensor) -> Tensor:
        truncated_mask = truncate_attention_mask(mask, self.right_size, self.left_size)
        return truncated_mask

    def _mask(self, att: Tensor, query_mask: Tensor, *args, **kwargs) -> Tensor:
        query_mask = query_mask.unsqueeze(dim=1)
        return att.masked_fill(~query_mask, self.masking_value)

    def forward(
        self,
        x: Tensor,
        mask: Union[Tensor, None],
    ) -> Tensor:
        """Applies truncated masked multi-head self attention to the input.

        Args:

            x (Tensor): The input tensor of shape [B, M, d].

            mask (Union[Tensor, None]): The mask tensor of the input of shape
            [B, M] where True indicates that the corresponding input position
            contains data not padding and therefore should not be masked.
            If None, the function will act as a normal multi-head self attention.

        Returns:

            Tensor: The attention result tensor of shape [B, M, d].

        """
        query_mask = None
        if mask is not None:
            query_mask = self.get_looking_ahead_mask(mask=mask)
        return super().forward(
            key=x, query=x, value=x, key_mask=mask, query_mask=query_mask
        )


class TruncatedRelativeMHSA(TruncatedSelfAttention):
    """Builds the truncated self attention with relative positional encoding
    module proposed in https://arxiv.org/abs/2002.02562

    Args:

        d_model (int): The model dimension.

        h (int): The number of attention heads.

        left_size (int): The size of the left window that each time step is
        allowed to look at.

        right_size (int): The size of the right window that each time step is
        allowed to look at.

        masking_value (float): The attention masking value.
    """

    def __init__(
        self,
        d_model: int,
        h: int,
        left_size: int,
        right_size: int,
        masking_value: float = -1e15,
    ) -> None:
        super().__init__(
            d_model=d_model,
            h=h,
            left_size=left_size,
            right_size=right_size,
            masking_value=masking_value,
        )

    def forward(
        self,
        x: Tensor,
        mask: Union[Tensor, None],
    ) -> Tensor:
        """Applies truncated masked rekative multi-head self attention to the input.

        Args:

            x (Tensor): The input tensor of shape [B, M, d].

            mask (Union[Tensor, None]): The mask tensor of the input of shape
            [B, M] where True indicates that the corresponding input position
            contains data not padding and therefore should not be masked.
            If None, the function will act as a normal multi-head self attention.

        Returns:

            Tensor: The attention result tensor of shape [B, M, d].

        """
        x = add_pos_enc(x)
        return super().forward(x=x, mask=mask)
