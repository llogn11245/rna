import torch
from torch import Tensor
import torch.nn as nn
from typing import List, Optional, Tuple, Union
import math

def get_mask_from_lens(lengths: Tensor, max_len: int) -> Tensor:
    """Creates a mask tensor from lengths tensor.

    Args:
        lengths (Tensor): The lengths of the original tensors of shape [B].

        max_len (int): the maximum lengths.

    Returns:
        Tensor: The mask of shape [B, max_len] and True whenever the index in the data portion.
    """
    indices = torch.arange(max_len).to(lengths.device)
    indices = indices.expand(len(lengths), max_len)
    return indices < lengths.unsqueeze(dim=1)

def truncate_attention_mask(mask: Tensor, right_size: int, left_size: int) -> Tensor:
    """creates a truncation mask that can be used to mask attention to only look
    at the time steps with a certain range. Specifically, it allows attention
    to look at right_size steps to the right and left_size steps to the left of
    each time step.


    Args:

        mask (Tensor): The original mask, which is True for the data positions
        and False for the padding ones. It has a shape of [B, M].

        right_size (int): The size of the right window that each time step is
        allowed to look at.

        left_size (int): The size of the left window that each time step is
        allowed to look at.


    Returns:
        Tensor: The new mask tensor of shape [B, M, M]
    """
    max_len = mask.shape[1]
    window_size = right_size + left_size + 1
    new_mask = torch.zeros(max_len**2, dtype=torch.bool).to(mask.device)
    # creating the original positions that will be the center of the window
    centers = torch.arange(0, max_len, device=mask.device)

    # the start and the end of each window
    start = torch.clamp_min(centers - left_size, 0)
    end = torch.clamp_max(centers + right_size, max_len - 1)

    # defining the indices in each window
    indices = (
        torch.arange(0, window_size, device=mask.device)
        .repeat(max_len)
        .view(max_len, -1)
    )
    indices = torch.clamp_max(start.view(-1, 1) + indices, end.view(-1, 1))
    indices += (torch.arange(0, max_len, device=mask.device) * max_len).view(-1, 1)
    
    # setting the indices to True
    new_mask = new_mask.index_put((indices,), torch.tensor(True)).view(max_len, max_len)
    # merging the original tensor with the new one
    return mask.unsqueeze(dim=1) & new_mask.unsqueeze(dim=0) & mask.unsqueeze(dim=-1)

class AddAndNorm(nn.Module):
    """Implements the Add and Norm module of the transformer model as described
    in the paper https://arxiv.org/abs/1706.03762

    Args:

        d_model (int): The model dimensionality.

    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.lnorm = nn.LayerNorm(normalized_shape=d_model)

    def forward(self, x: Tensor, sub_x: Tensor) -> Tensor:
        """takes the output tensor `x` from the last layer and the output tensor
        `sub_x` from the sub-layer, adds them, and then normalizes the sum
        using layer normalization.

        Args:
            x (Tensor): The output tensor of the last layer with shape [B, M, d].

            sub_x (Tensor): The output tensor of the sub-layer with shape
            [B, M, d].

        Returns:
            Tensor: The result tensor obtained after normalizing the sum of
            the inputs with shape [B, M, d].

        """
        return self.lnorm(x + sub_x)


class FeedForwardModule(nn.Module):
    """Implements the feed-forward module of the transformer architecture as
    described in the paper https://arxiv.org/abs/1706.03762

    Args:
        d_model (int): The model dimensionality.

        ff_size (int): The dimensionality of the inner layer.
    """

    def __init__(self, d_model: int, ff_size: int, p_dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features=d_model, out_features=ff_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=ff_size, out_features=d_model)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Passes the input to the layer

        Args:
            x (Tensor): The input tensor of shape [B, M, d]

        Returns:
            Tensor: The output tensor of shape [B, M, d] obtained after passing
            through the layer.
        """
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        return out

class CausalVGGBlock(nn.Module):
    """Implements a causal VGG block consisting of causal 2D convolution layers,
    as described in the paper https://arxiv.org/pdf/1910.12977.pdf.



    Args:
        n_conv (int): Specifies the number of convolution layers.

        in_channels (int): Specifies the number of input channels.

        out_channels (List[int]): A list of integers that specifies the number
        of channels in each convolution layer

        kernel_sizes (List[int]): A list of integers that specifies the kernel size of each convolution layer.

        pooling_kernel_size (int): Specifies the kernel size of the pooling layer.

    """

    def __init__(
        self,
        n_conv: int,
        in_channels: int,
        out_channels: List[int],
        kernel_sizes: List[int],
        pooling_kernel_size: int,
    ) -> None:
        super().__init__()
        self.conv_layers = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_channels if i == 0 else out_channels[i - 1],
                    out_channels=out_channels[i],
                    kernel_size=kernel_sizes[i],
                )
                for i in range(n_conv)
            ]
        )
        self.pooling = nn.MaxPool2d(kernel_size=pooling_kernel_size)

    def _pad(self, x: Tensor, kernel_size: Tuple[int, int]):
        batch_size, channels, max_len, feat_size = x.shape
        seq_pad = torch.zeros(batch_size, channels, kernel_size[0] - 1, feat_size).to(
            x.device
        )
        feat_pad = torch.zeros(
            batch_size, channels, kernel_size[0] - 1 + max_len, kernel_size[1] - 1
        ).to(x.device)
        x = torch.cat([seq_pad, x], dim=2)
        x = torch.cat([feat_pad, x], dim=3)
        return x

    def forward(self, x: Tensor, lengths: Tensor) -> Tuple[Tensor, Tensor]:
        """passes the input x of shape [B, C, M, f] to the network.

        Args:
            x (Tensor): The input tensor if shape [B, C, M, f].
            lengths (Tensor): The legnths tensor of shape [B].

        Returns:
            Tuple[Tensor, Tensor]: A tuple where the first is the result of shape
            [B, C', M', f'] and the updated lengths of shape [B]
        """
        for conv_layer in self.conv_layers:
            kernel_size = conv_layer.kernel_size
            x = self._pad(x, kernel_size=kernel_size)
            x = conv_layer(x)
        x = self.pooling(x)
        lengths = lengths // self.pooling.kernel_size
        return x, lengths


class VGGTransformerPreNet(nn.Module):
    """Implements the VGGTransformer prenet module as described in
    https://arxiv.org/abs/1910.12977

    Args:

    in_features (int): The input feature size.

    n_vgg_blocks (int): The number of VGG blocks to use.

    n_layers_per_block (List[int]): A list of integers that specifies the number
    of convolution layers in each block.

    kernel_sizes_per_block (List[List[int]]): A list of lists that contains the
    kernel size for each layer in each block. The length of the outer list
    should match `n_vgg_blocks`, and each inner list should be the same length
    as the corresponding block's number of layers.

    n_channels_per_block (List[List[int]]): A list of lists that contains the
    number of channels for each convolution layer in each block. This argument
    should also have length equal to `n_vgg_blocks`, and each sublist should
    have length equal to the number of layers in the corresponding block.

    pooling_kernel_size (List[int]): A list of integers that specifies the size
    of the max pooling layer in each block. The length of this list should be
    equal to `n_vgg_blocks`.

    d_model (int): The size of the output feature

    """

    def __init__(
        self,
        in_features: int,
        n_vgg_blocks: int,
        n_layers_per_block: List[int],
        kernel_sizes_per_block: List[List[int]],
        n_channels_per_block: List[List[int]],
        pooling_kernel_size: List[int],
        d_model: int,
    ) -> None:
        super().__init__()
        self.vgg_blocks = nn.ModuleList(
            [
                CausalVGGBlock(
                    n_conv=n_layers_per_block[i],
                    in_channels=1 if i == 0 else n_channels_per_block[i - 1][-1],
                    out_channels=n_channels_per_block[i],
                    kernel_sizes=kernel_sizes_per_block[i],
                    pooling_kernel_size=pooling_kernel_size[i],
                )
                for i in range(n_vgg_blocks)
            ]
        )
        for i in range(n_vgg_blocks):
            in_features //= pooling_kernel_size[i]
        in_features *= n_channels_per_block[-1][-1]
        self.fc = nn.Linear(in_features=in_features, out_features=d_model)

    def forward(self, x: Tensor, lengths: Tensor) -> Tuple[Tensor, Tensor]:
        """Passes the input `x` through the VGGTransformer prenet and returns
        a tuple of tensors.

        Args:
            x (Tensor): Input tensor of shape [B, M, in_features].

            lengths (Tensor): Lengths of shape [B] that has the length for each
            sequence in `x`.

        Returns:
            A tuple of tensors (output, updated_lengths).
            - output (Tensor): Output tensor of shape [B, M, d_model].
            - updated_lengths (Tensor): Updated lengths of shape [B].
        """
        x = x.unsqueeze(dim=1)  # [B, 1, M, d]
        for block in self.vgg_blocks:
            x, lengths = block(x, lengths)
        x = x.permute(0, 2, 1, 3)
        x = x.contiguous()
        x = x.view(*x.shape[:2], -1)
        return self.fc(x), lengths

def get_positional_encoding(max_length: int, d_model: int) -> Tensor:
    """Create positional encoding tensor as described in
    https://arxiv.org/abs/1706.03762

    Args:

        max_length (int): The maximum length of the positionals sequence.

        d_model (int): The dimensionality of the positionals sequence.

    Returns:

        Tensor: Positional tensor of shape [1, max_length, d_model]

    """
    if d_model % 2 == 1:
        raise ValueError("Even number is expected for d_model, but odd is given!")
    result = torch.zeros(max_length, d_model, dtype=torch.float)
    feat_range = torch.arange(0, d_model // 2)
    time_range = torch.arange(0, max_length)
    denominator = pow(10000, 2 * feat_range / d_model)
    result[:, 0::2] = torch.sin(time_range[:, None] / denominator)
    result[:, 1::2] = torch.cos(time_range[:, None] / denominator)
    result = result.unsqueeze(dim=0)
    return result

def add_pos_enc(x: Tensor) -> Tensor:
    """Adds positional encodings to the input tensor x.

    Args:

        x (Tensor): The input tensor of shape [B, M, d].

    Returns:

        Tensor: The input added to at the positional encoding.

    """
    d_model = x.shape[-1]
    pe = get_positional_encoding(x.shape[1], d_model)
    pe = pe.to(x.device)
    return pe + x


def calc_data_len(
    result_len: int,
    pad_len: Union[Tensor, int],
    data_len: Union[Tensor, int],
    kernel_size: int,
    stride: int,
) -> Union[Tensor, int]:
    """Calculates the new data portion size after applying convolution on a padded tensor

    Args:

        result_len (int): The length after the convolution is applied.

        pad_len Union[Tensor, int]: The original padding portion length.

        data_len Union[Tensor, int]: The original data portion legnth.

        kernel_size (int): The convolution kernel size.

        stride (int): The convolution stride.

    Returns:

        Union[Tensor, int]: The new data portion length.

    """
    if type(pad_len) != type(data_len):
        raise ValueError(
            f"""expected both pad_len and data_len to be of the same type
            but {type(pad_len)}, and {type(data_len)} passed"""
        )
    inp_len = data_len + pad_len
    new_pad_len = 0
    # if padding size less than the kernel size
    # then it will be convolved with the data.
    convolved_pad_mask = pad_len >= kernel_size
    # calculating the size of the discarded items (not convolved)
    unconvolved = (inp_len - kernel_size) % stride
    undiscarded_pad_mask = unconvolved < pad_len
    convolved = pad_len - unconvolved
    new_pad_len = (convolved - kernel_size) // stride + 1
    # setting any condition violation to zeros using masks
    new_pad_len *= convolved_pad_mask
    new_pad_len *= undiscarded_pad_mask
    return result_len - new_pad_len

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, T, D] --> x + pe[:, :T, :]
        return x + self.pe[:, : x.size(1), :]
