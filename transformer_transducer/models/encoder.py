from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .attention import MultiHeadAtt, TruncatedSelfAttention, TruncatedRelativeMHSA
from .utils import AddAndNorm, FeedForwardModule, get_mask_from_lens, VGGTransformerPreNet, calc_data_len, add_pos_enc

class TransformerEncLayer(nn.Module):
    """Implements a single layer of the transformer encoder model as
    presented in the paper https://arxiv.org/abs/1706.03762

    Args:
        d_model (int): The model dimensionality.

        ff_size (int): The feed forward inner layer dimensionality.

        h (int): The number of heads in the attention mechanism.

        masking_value (float, optional): The value to use for masking padded
        elements. Defaults to -1e15.
    """

    def __init__(
        self, 
        d_model: int, 
        ff_size: int, 
        h: int, 
        p_dropout: float,
        masking_value: int = -1e15,
    ) -> None:
        super().__init__()
        self.mhsa = MultiHeadAtt(d_model=d_model, h=h, masking_value=masking_value)
        self.add_and_norm1 = AddAndNorm(d_model=d_model)
        self.ff = FeedForwardModule(d_model=d_model, ff_size=ff_size, p_dropout=p_dropout)
        self.add_and_norm2 = AddAndNorm(d_model=d_model)
        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, x: Tensor, mask: Union[Tensor, None] = None) -> Tensor:
        """Performs a forward pass of the transformer encoder layer.

        Args:

            x (Tensor): The input tensor of shape [B, M, d].

            mask (Union[Tensor, None], optional): Boolean tensor of the input of shape
            [B, M] where True indicates that the corresponding key position
            contains data not padding and therefore should not be masked.
            If None, the function will act as a normal multi-head attention. Defaults to None.

        Returns:
            Tensor: Result tensor of the same shape as x.
        """
        # out = self.mhsa(key=x, query=x, value=x, key_mask=mask, query_mask=mask)
        # out = self.add_and_norm1(x, out)
        # ff = self.ff(out)
        # result = self.add_and_norm2(out,ff)
        #         
        # return result

        att = self.mhsa(key=x, query=x, value=x, key_mask=mask, query_mask=mask)
        att = self.dropout(att)
        out = self.add_and_norm1(x, att)

        ff  = self.ff(out)
        ff  = self.dropout(ff)

        return self.add_and_norm2(out, ff)

class TransformerEncLayerWithAttTruncation(TransformerEncLayer):
    """Implements a single encoder layer of the transformer
    with truncated self attention as described in https://arxiv.org/abs/1910.12977

    Args:

        d_model (int): The model dimensionality.

        ff_size (int): The feed forward inner layer dimensionality.

        h (int): The number of heads in the attention mechanism.

        left_size (int): The size of the left window that each time step is
        allowed to look at.

        right_size (int): The size of the right window that each time step is
        allowed to look at.

        masking_value (float, optional): The value to use for masking padded
        elements. Defaults to -1e15.
    """

    def __init__(
        self,
        d_model: int,
        ff_size: int,
        h: int,
        left_size: int,
        right_size: int,
        masking_value: int = -1e15,
    ) -> None:
        super().__init__(
            d_model=d_model, ff_size=ff_size, h=h, masking_value=masking_value
        )
        self.mhsa = TruncatedSelfAttention(
            d_model=d_model,
            h=h,
            left_size=left_size,
            right_size=right_size,
            masking_value=masking_value,
        )

    def forward(self, x: Tensor, mask: Union[Tensor, None] = None) -> Tensor:
        """Performs a forward pass of the transformer encoder layer.

        Args:

            x (Tensor): The input tensor of shape [B, M, d].

            mask (Union[Tensor, None], optional): Boolean tensor of the input of shape
            [B, M] where True indicates that the corresponding key position
            contains data not padding and therefore should not be masked.
            If None, the function will act as a normal multi-head attention. Defaults to None.

        Returns:
            Tensor: Result tensor of the same shape as x.
        """
        out = self.mhsa(x=x, mask=mask)
        out = self.add_and_norm1(x, out)
        result = self.ff(out)
        return self.add_and_norm2(out, result)


class VGGTransformerEncoder(nn.Module):
    """Implements the VGGTransformer encoder as described in
    https://arxiv.org/abs/1910.12977

    Args:

        in_features (int): The input feature size.

        n_layers (int): The number of transformer encoder layers with truncated
        self attention.

        n_vgg_blocks (int): The number of VGG blocks to use.

        n_conv_layers_per_vgg_block (List[int]): A list of integers that specifies the number
        of convolution layers in each block.

        kernel_sizes_per_vgg_block (List[List[int]]): A list of lists that contains the
        kernel size for each layer in each block. The length of the outer list
        should match `n_vgg_blocks`, and each inner list should be the same length
        as the corresponding block's number of layers.

        n_channels_per_vgg_block (List[List[int]]): A list of lists that contains the
        number of channels for each convolution layer in each block. This argument
        should also have length equal to `n_vgg_blocks`, and each sublist should
        have length equal to the number of layers in the corresponding block.

        vgg_pooling_kernel_size (List[int]): A list of integers that specifies the size
        of the max pooling layer in each block. The length of this list should be
        equal to `n_vgg_blocks`.

        d_model (int): The model dimensionality.

        ff_size (int): The feed forward inner layer dimensionality.

        h (int): The number of heads in the attention mechanism.

        left_size (int): The size of the left window that each time step is
        allowed to look at.

        right_size (int): The size of the right window that each time step is
        allowed to look at.

        masking_value (float, optional): The value to use for masking padded
        elements. Defaults to -1e15.
    """

    def __init__(
        self,
        in_features: int,
        n_layers: int,
        n_vgg_blocks: int,
        n_conv_layers_per_vgg_block: List[int],
        kernel_sizes_per_vgg_block: List[List[int]],
        n_channels_per_vgg_block: List[List[int]],
        vgg_pooling_kernel_size: List[int],
        d_model: int,
        ff_size: int,
        h: int,
        left_size: int,
        right_size: int,
        masking_value: int = -1e15,
    ) -> None:
        super().__init__()
        self.pre_net = VGGTransformerPreNet(
            in_features=in_features,
            n_vgg_blocks=n_vgg_blocks,
            n_layers_per_block=n_conv_layers_per_vgg_block,
            kernel_sizes_per_block=kernel_sizes_per_vgg_block,
            n_channels_per_block=n_channels_per_vgg_block,
            pooling_kernel_size=vgg_pooling_kernel_size,
            d_model=d_model,
        )
        self.enc_layers = nn.ModuleList(
            [
                TransformerEncLayerWithAttTruncation(
                    d_model=d_model,
                    ff_size=ff_size,
                    h=h,
                    left_size=left_size,
                    right_size=right_size,
                    masking_value=masking_value,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        """Passes the input `x` through the encoder layers.

        Args:

            x (Tensor): The input speech tensor of shape [B, M, d]

            mask (Tensor): The input boolean mask of shape [B, M], where it's True
            if there is no padding.

        Returns:

            Tuple[Tensor, Tensor]: A tuple where the first element is the encoded speech of shape
            [B, M, F] and the second element is the lengths of shape [B].
        """
        lengths = mask.sum(dim=-1)
        out, lengths = self.pre_net(x, lengths)
        mask = get_mask_from_lens(lengths=lengths, max_len=out.shape[1])
        for layer in self.enc_layers:
            out = layer(out, mask)
        return out, lengths


class TransformerTransducerLayer(nn.Module):
    """Implements a single encoder layer of the transformer transducer
    with truncated relative self attention as described in https://arxiv.org/abs/2002.02562

    Args:

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
        d_model: int,
        ff_size: int,
        h: int,
        left_size: int,
        right_size: int,
        p_dropout: float,
        masking_value: int = -1e15,
    ) -> None:
        super().__init__()
        self.mhsa = TruncatedRelativeMHSA(
            d_model=d_model,
            h=h,
            left_size=left_size,
            right_size=right_size,
            masking_value=masking_value,
        )
        self.add_and_norm = AddAndNorm(d_model=d_model)
        self.ff = FeedForwardModule(
            d_model=d_model, ff_size=ff_size, p_dropout=p_dropout
        )
        self.dropout = nn.Dropout(p_dropout)
        self.lnorm = nn.LayerNorm(normalized_shape=d_model)

    def forward(self, x: Tensor, mask: Union[Tensor, None] = None) -> Tensor:
        """Performs a forward pass of the transformer-transducer encoder layer.

        Args:

            x (Tensor): The input tensor of shape [B, M, d].

            mask (Union[Tensor, None], optional): Boolean tensor of the input of shape
            [B, M] where True indicates that the corresponding key position
            contains data not padding and therefore should not be masked.
            If None, the function will act as a normal multi-head attention. Defaults to None.

        Returns:
            Tensor: Result tensor of the same shape as x.
        """
        x = self.lnorm(x)
        out = self.mhsa(x, mask)
        out = self.add_and_norm(x, out)
        out = out + self.ff(out)
        out = self.dropout(out)
        return out

class TransformerTransducerEncoder(nn.Module):
    """Implements the Transformer-Transducer encoder with relative truncated
    multi-head self attention as described in https://arxiv.org/abs/2002.02562

    Args:

        in_features (int): The input feature size.

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

        stride (int): The stride of the convolution layer. Default 1.

        kernel_size (int): The kernel size of the convolution layer. Default 1.

        masking_value (float, optional): The value to use for masking padded
        elements. Defaults to -1e15.
    """

    def __init__(
        self,
        in_features: int,
        n_layers: int,
        d_model: int,
        ff_size: int,
        h: int,
        left_size: int,
        right_size: int,
        p_dropout: float,
        stride: int = 1,
        kernel_size: int = 1,
        masking_value: int = -1e15,
    ) -> None:
        super().__init__()
        self.pre_net = nn.Conv1d(
            in_channels=in_features,
            out_channels=d_model,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.enc_layers = nn.ModuleList(
            [
                # TransformerTransducerLayer(
                #     d_model=d_model,
                #     ff_size=ff_size,
                #     h=h,
                #     left_size=left_size,
                #     right_size=right_size,
                #     p_dropout=p_dropout,
                #     masking_value=masking_value,
                # )
                TransformerEncLayer(
                    d_model=d_model,
                    ff_size=ff_size,
                    h=h,
                    p_dropout=p_dropout,
                    masking_value=masking_value,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        """Passes the input `x` through the encoder layers.

        Args:

            x (Tensor): The input speech tensor of shape [B, M, d]

            mask (Tensor): The input boolean mask of shape [B, M], where it's True
            if there is no padding.

        Returns:

            Tuple[Tensor, Tensor]: A tuple where the first element is the encoded speech of shape
            [B, M, F] and the second element is the lengths of shape [B].
        """
        lengths = mask.sum(dim=-1)
        out = x.transpose(-1, -2)
        out = self.pre_net(out)
        out = out.transpose(-1, -2)

        out = add_pos_enc(out)

        lengths = calc_data_len(
            result_len=out.shape[1],
            pad_len=x.shape[1] - lengths,
            data_len=lengths,
            kernel_size=self.pre_net.kernel_size[0],
            stride=self.pre_net.stride[0],
        )
        mask = get_mask_from_lens(lengths=lengths, max_len=out.shape[1])
        for layer in self.enc_layers:
            out = layer(out, mask)
        return out, lengths