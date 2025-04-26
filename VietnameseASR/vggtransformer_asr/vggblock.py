# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function, unicode_literals

from collections.abc import Iterable
from itertools import repeat

import torch
import torch.nn as nn
# from fairseq import utils

def _pair(v):
    if isinstance(v, Iterable):
        assert len(v) == 2, "len(v) != 2"
        return v
    return tuple(repeat(v, 2))


def infer_conv_output_dim(conv_op, input_dim, sample_inchannel):
    sample_seq_len = 200
    sample_bsz = 10
    x = torch.randn(sample_bsz, sample_inchannel, sample_seq_len, input_dim)
    # N x C x H x W
    # N: sample_bsz, C: sample_inchannel, H: sample_seq_len, W: input_dim
    x = conv_op(x)
    # N x C x H x W
    x = x.transpose(1, 2)
    # N x H x C x W
    bsz, seq = x.size()[:2]
    per_channel_dim = x.size()[3]
    # bsz: N, seq: H, CxW the rest
    return x.contiguous().view(bsz, seq, -1).size(-1), per_channel_dim


class VGGBlock(torch.nn.Module):
    """
    VGG motibated cnn module https://arxiv.org/pdf/1409.1556.pdf

    Args:
        in_channels: (int) number of input channels (typically 1)
        out_channels: (int) number of output channels
        conv_kernel_size: convolution channels
        pooling_kernel_size: the size of the pooling window to take a max over
        num_conv_layers: (int) number of convolution layers
        input_dim: (int) input dimension
        conv_stride: the stride of the convolving kernel.
            Can be a single number or a tuple (sH, sW)  Default: 1
        padding: implicit paddings on both sides of the input.
            Can be a single number or a tuple (padH, padW). Default: None
        layer_norm: (bool) if layer norm is going to be applied. Default: False

    Shape:
        Input: BxCxTxfeat, i.e. (batch_size, input_size, timesteps, features)
        Output: BxCxTxfeat, i.e. (batch_size, input_size, timesteps, features)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        conv_kernel_size,
        pooling_kernel_size,
        num_conv_layers,
        input_dim,
        conv_stride=1,
        padding=None,
        layer_norm=False,
    ):
        assert (
            input_dim is not None
        ), "Need input_dim for LayerNorm and infer_conv_output_dim"
        super(VGGBlock, self).__init__()
        self.in_channels = in_channels # 64
        self.out_channels = out_channels # 3
        self.conv_kernel_size = _pair(conv_kernel_size) # 2 
        self.pooling_kernel_size = _pair(pooling_kernel_size) # 2 
        self.num_conv_layers = num_conv_layers 
        self.padding = (
            tuple(e // 2 for e in self.conv_kernel_size)
            if padding is None
            else _pair(padding)
        )
        self.conv_stride = _pair(conv_stride)

        self.layers = nn.ModuleList()
        for layer in range(num_conv_layers):
            conv_op = nn.Conv2d(
                in_channels if layer == 0 else out_channels,
                out_channels,
                self.conv_kernel_size,
                stride=self.conv_stride,
                padding=self.padding,
            )
            self.layers.append(conv_op)
            if layer_norm:
                conv_output_dim, per_channel_dim = infer_conv_output_dim(
                    conv_op, input_dim, in_channels if layer == 0 else out_channels
                )
                self.layers.append(nn.LayerNorm(per_channel_dim))
                input_dim = per_channel_dim
            self.layers.append(nn.ReLU())

        if self.pooling_kernel_size is not None:
            pool_op = nn.MaxPool2d(kernel_size=self.pooling_kernel_size, ceil_mode=True)
            self.layers.append(pool_op)
            self.total_output_dim, self.output_dim = infer_conv_output_dim(
                pool_op, input_dim, out_channels
            )

    def forward(self, x):
        for i, _ in enumerate(self.layers):
            x = self.layers[i](x)
        return x



# class VGGTransformerEncoder(FairseqEncoder):
#     """VGG + Transformer encoder"""

#     def __init__(
#         self,
#         input_feat_per_channel,
#         vggblock_config=DEFAULT_ENC_VGGBLOCK_CONFIG,
#         transformer_config=DEFAULT_ENC_TRANSFORMER_CONFIG,
#         encoder_output_dim=512,
#         in_channels=1,
#         transformer_context=None,
#         transformer_sampling=None,
#     ):
#         """constructor for VGGTransformerEncoder

#         Args:
#             - input_feat_per_channel: feature dim (not including stacked,
#               just base feature)
#             - in_channel: # input channels (e.g., if stack 8 feature vector
#                 together, this is 8)
#             - vggblock_config: configuration of vggblock, see comments on
#                 DEFAULT_ENC_VGGBLOCK_CONFIG
#             - transformer_config: configuration of transformer layer, see comments
#                 on DEFAULT_ENC_TRANSFORMER_CONFIG
#             - encoder_output_dim: final transformer output embedding dimension
#             - transformer_context: (left, right) if set, self-attention will be focused
#               on (t-left, t+right)
#             - transformer_sampling: an iterable of int, must match with
#               len(transformer_config), transformer_sampling[i] indicates sampling
#               factor for i-th transformer layer, after multihead att and feedfoward
#               part
#         """
#         super().__init__(None)

#         self.num_vggblocks = 0
#         if vggblock_config is not None:
#             if not isinstance(vggblock_config, Iterable):
#                 raise ValueError("vggblock_config is not iterable")
#             self.num_vggblocks = len(vggblock_config)

#         self.conv_layers = nn.ModuleList()
#         self.in_channels = in_channels
#         self.input_dim = input_feat_per_channel
#         self.pooling_kernel_sizes = []

#         if vggblock_config is not None:
#             for _, config in enumerate(vggblock_config):
#                 (
#                     out_channels,
#                     conv_kernel_size,
#                     pooling_kernel_size,
#                     num_conv_layers,
#                     layer_norm,
#                 ) = config
#                 self.conv_layers.append(
#                     VGGBlock(
#                         in_channels,
#                         out_channels,
#                         conv_kernel_size,
#                         pooling_kernel_size,
#                         num_conv_layers,
#                         input_dim=input_feat_per_channel,
#                         layer_norm=layer_norm,
#                     )
#                 )
#                 self.pooling_kernel_sizes.append(pooling_kernel_size)
#                 in_channels = out_channels
#                 input_feat_per_channel = self.conv_layers[-1].output_dim

#         transformer_input_dim = self.infer_conv_output_dim(
#             self.in_channels, self.input_dim
#         )
#         # transformer_input_dim is the output dimension of VGG part

#         self.validate_transformer_config(transformer_config)
#         self.transformer_context = self.parse_transformer_context(transformer_context)
#         self.transformer_sampling = self.parse_transformer_sampling(
#             transformer_sampling, len(transformer_config)
#         )

#         self.transformer_layers = nn.ModuleList()

#         if transformer_input_dim != transformer_config[0][0]:
#             self.transformer_layers.append(
#                 Linear(transformer_input_dim, transformer_config[0][0])
#             )
#         self.transformer_layers.append(
#             TransformerEncoderLayer(
#                 prepare_transformer_encoder_params(*transformer_config[0])
#             )
#         )

#         for i in range(1, len(transformer_config)):
#             if transformer_config[i - 1][0] != transformer_config[i][0]:
#                 self.transformer_layers.append(
#                     Linear(transformer_config[i - 1][0], transformer_config[i][0])
#                 )
#             self.transformer_layers.append(
#                 TransformerEncoderLayer(
#                     prepare_transformer_encoder_params(*transformer_config[i])
#                 )
#             )

#         self.encoder_output_dim = encoder_output_dim
#         self.transformer_layers.extend(
#             [
#                 Linear(transformer_config[-1][0], encoder_output_dim),
#                 LayerNorm(encoder_output_dim),
#             ]
#         )

#     def forward(self, src_tokens, src_lengths, **kwargs):
#         """
#         src_tokens: padded tensor (B, T, C * feat)
#         src_lengths: tensor of original lengths of input utterances (B,)
#         """
#         bsz, max_seq_len, _ = src_tokens.size()
#         x = src_tokens.view(bsz, max_seq_len, self.in_channels, self.input_dim)
#         x = x.transpose(1, 2).contiguous()
#         # (B, C, T, feat)

#         for layer_idx in range(len(self.conv_layers)):
#             x = self.conv_layers[layer_idx](x)

#         bsz, _, output_seq_len, _ = x.size()

#         # (B, C, T, feat) -> (B, T, C, feat) -> (T, B, C, feat) -> (T, B, C * feat)
#         x = x.transpose(1, 2).transpose(0, 1)
#         x = x.contiguous().view(output_seq_len, bsz, -1)

#         input_lengths = src_lengths.clone()
#         for s in self.pooling_kernel_sizes:
#             input_lengths = (input_lengths.float() / s).ceil().long()

#         encoder_padding_mask, _ = lengths_to_encoder_padding_mask(
#             input_lengths, batch_first=True
#         )
#         if not encoder_padding_mask.any():
#             encoder_padding_mask = None

#         subsampling_factor = int(max_seq_len * 1.0 / output_seq_len + 0.5)
#         attn_mask = self.lengths_to_attn_mask(input_lengths, subsampling_factor)

#         transformer_layer_idx = 0

#         for layer_idx in range(len(self.transformer_layers)):

#             if isinstance(self.transformer_layers[layer_idx], TransformerEncoderLayer):
#                 x = self.transformer_layers[layer_idx](
#                     x, encoder_padding_mask, attn_mask
#                 )

#                 if self.transformer_sampling[transformer_layer_idx] != 1:
#                     sampling_factor = self.transformer_sampling[transformer_layer_idx]
#                     x, encoder_padding_mask, attn_mask = self.slice(
#                         x, encoder_padding_mask, attn_mask, sampling_factor
#                     )

#                 transformer_layer_idx += 1

#             else:
#                 x = self.transformer_layers[layer_idx](x)

#         # encoder_padding_maks is a (T x B) tensor, its [t, b] elements indicate
#         # whether encoder_output[t, b] is valid or not (valid=0, invalid=1)

#         return {
#             "encoder_out": x,  # (T, B, C)
#             "encoder_padding_mask": encoder_padding_mask.t()
#             if encoder_padding_mask is not None
#             else None,
#             # (B, T) --> (T, B)
#         }

#     def infer_conv_output_dim(self, in_channels, input_dim):
#         sample_seq_len = 200
#         sample_bsz = 10
#         x = torch.randn(sample_bsz, in_channels, sample_seq_len, input_dim)
#         for i, _ in enumerate(self.conv_layers):
#             x = self.conv_layers[i](x)
#         x = x.transpose(1, 2)
#         mb, seq = x.size()[:2]
#         return x.contiguous().view(mb, seq, -1).size(-1)

#     def validate_transformer_config(self, transformer_config):
#         for config in transformer_config:
#             input_dim, num_heads = config[:2]
#             if input_dim % num_heads != 0:
#                 msg = (
#                     "ERROR in transformer config {}: ".format(config)
#                     + "input dimension {} ".format(input_dim)
#                     + "not dividable by number of heads {}".format(num_heads)
#                 )
#                 raise ValueError(msg)

#     def parse_transformer_context(self, transformer_context):
#         """
#         transformer_context can be the following:
#         -   None; indicates no context is used, i.e.,
#             transformer can access full context
#         -   a tuple/list of two int; indicates left and right context,
#             any number <0 indicates infinite context
#                 * e.g., (5, 6) indicates that for query at x_t, transformer can
#                 access [t-5, t+6] (inclusive)
#                 * e.g., (-1, 6) indicates that for query at x_t, transformer can
#                 access [0, t+6] (inclusive)
#         """
#         if transformer_context is None:
#             return None

#         if not isinstance(transformer_context, Iterable):
#             raise ValueError("transformer context must be Iterable if it is not None")

#         if len(transformer_context) != 2:
#             raise ValueError("transformer context must have length 2")

#         left_context = transformer_context[0]
#         if left_context < 0:
#             left_context = None

#         right_context = transformer_context[1]
#         if right_context < 0:
#             right_context = None

#         if left_context is None and right_context is None:
#             return None

#         return (left_context, right_context)

#     def parse_transformer_sampling(self, transformer_sampling, num_layers):
#         """
#         parsing transformer sampling configuration

#         Args:
#             - transformer_sampling, accepted input:
#                 * None, indicating no sampling
#                 * an Iterable with int (>0) as element
#             - num_layers, expected number of transformer layers, must match with
#               the length of transformer_sampling if it is not None

#         Returns:
#             - A tuple with length num_layers
#         """
#         if transformer_sampling is None:
#             return (1,) * num_layers

#         if not isinstance(transformer_sampling, Iterable):
#             raise ValueError(
#                 "transformer_sampling must be an iterable if it is not None"
#             )

#         if len(transformer_sampling) != num_layers:
#             raise ValueError(
#                 "transformer_sampling {} does not match with the number "
#                 "of layers {}".format(transformer_sampling, num_layers)
#             )

#         for layer, value in enumerate(transformer_sampling):
#             if not isinstance(value, int):
#                 raise ValueError("Invalid value in transformer_sampling: ")
#             if value < 1:
#                 raise ValueError(
#                     "{} layer's subsampling is {}.".format(layer, value)
#                     + " This is not allowed! "
#                 )
#         return transformer_sampling

#     def slice(self, embedding, padding_mask, attn_mask, sampling_factor):
#         """
#         embedding is a (T, B, D) tensor
#         padding_mask is a (B, T) tensor or None
#         attn_mask is a (T, T) tensor or None
#         """
#         embedding = embedding[::sampling_factor, :, :]
#         if padding_mask is not None:
#             padding_mask = padding_mask[:, ::sampling_factor]
#         if attn_mask is not None:
#             attn_mask = attn_mask[::sampling_factor, ::sampling_factor]

#         return embedding, padding_mask, attn_mask

#     def lengths_to_attn_mask(self, input_lengths, subsampling_factor=1):
#         """
#         create attention mask according to sequence lengths and transformer
#         context

#         Args:
#             - input_lengths: (B, )-shape Int/Long tensor; input_lengths[b] is
#               the length of b-th sequence
#             - subsampling_factor: int
#                 * Note that the left_context and right_context is specified in
#                   the input frame-level while input to transformer may already
#                   go through subsampling (e.g., the use of striding in vggblock)
#                   we use subsampling_factor to scale the left/right context

#         Return:
#             - a (T, T) binary tensor or None, where T is max(input_lengths)
#                 * if self.transformer_context is None, None
#                 * if left_context is None,
#                     * attn_mask[t, t + right_context + 1:] = 1
#                     * others = 0
#                 * if right_context is None,
#                     * attn_mask[t, 0:t - left_context] = 1
#                     * others = 0
#                 * elsif
#                     * attn_mask[t, t - left_context: t + right_context + 1] = 0
#                     * others = 1
#         """
#         if self.transformer_context is None:
#             return None

#         maxT = torch.max(input_lengths).item()
#         attn_mask = torch.zeros(maxT, maxT)

#         left_context = self.transformer_context[0]
#         right_context = self.transformer_context[1]
#         if left_context is not None:
#             left_context = math.ceil(self.transformer_context[0] / subsampling_factor)
#         if right_context is not None:
#             right_context = math.ceil(self.transformer_context[1] / subsampling_factor)

#         for t in range(maxT):
#             if left_context is not None:
#                 st = 0
#                 en = max(st, t - left_context)
#                 attn_mask[t, st:en] = 1
#             if right_context is not None:
#                 st = t + right_context + 1
#                 st = min(st, maxT - 1)
#                 attn_mask[t, st:] = 1

#         return attn_mask.to(input_lengths.device)

#     def reorder_encoder_out(self, encoder_out, new_order):
#         encoder_out["encoder_out"] = encoder_out["encoder_out"].index_select(
#             1, new_order
#         )
#         if encoder_out["encoder_padding_mask"] is not None:
#             encoder_out["encoder_padding_mask"] = encoder_out[
#                 "encoder_padding_mask"
#             ].index_select(1, new_order)
#         return encoder_out


# class TransformerDecoder(FairseqIncrementalDecoder):
#     """
#     Transformer decoder consisting of *args.decoder_layers* layers. Each layer
#     is a :class:`TransformerDecoderLayer`.
#     Args:
#         args (argparse.Namespace): parsed command-line arguments
#         dictionary (~fairseq.data.Dictionary): decoding dictionary
#         embed_tokens (torch.nn.Embedding): output embedding
#         no_encoder_attn (bool, optional): whether to attend to encoder outputs.
#             Default: ``False``
#         left_pad (bool, optional): whether the input is left-padded. Default:
#             ``False``
#     """

#     def __init__(
#         self,
#         dictionary,
#         embed_dim=512,
#         transformer_config=DEFAULT_ENC_TRANSFORMER_CONFIG,
#         conv_config=DEFAULT_DEC_CONV_CONFIG,
#         encoder_output_dim=512,
#     ):

#         super().__init__(dictionary)
#         vocab_size = len(dictionary)
#         self.padding_idx = dictionary.pad()
#         self.embed_tokens = Embedding(vocab_size, embed_dim, self.padding_idx)

#         self.conv_layers = nn.ModuleList()
#         for i in range(len(conv_config)):
#             out_channels, kernel_size, layer_norm = conv_config[i]
#             if i == 0:
#                 conv_layer = LinearizedConv1d(
#                     embed_dim, out_channels, kernel_size, padding=kernel_size - 1
#                 )
#             else:
#                 conv_layer = LinearizedConv1d(
#                     conv_config[i - 1][0],
#                     out_channels,
#                     kernel_size,
#                     padding=kernel_size - 1,
#                 )
#             self.conv_layers.append(conv_layer)
#             if layer_norm:
#                 self.conv_layers.append(nn.LayerNorm(out_channels))
#             self.conv_layers.append(nn.ReLU())

#         self.layers = nn.ModuleList()
#         if conv_config[-1][0] != transformer_config[0][0]:
#             self.layers.append(Linear(conv_config[-1][0], transformer_config[0][0]))
#         self.layers.append(
#             TransformerDecoderLayer(
#                 prepare_transformer_decoder_params(*transformer_config[0])
#             )
#         )

#         for i in range(1, len(transformer_config)):
#             if transformer_config[i - 1][0] != transformer_config[i][0]:
#                 self.layers.append(
#                     Linear(transformer_config[i - 1][0], transformer_config[i][0])
#                 )
#             self.layers.append(
#                 TransformerDecoderLayer(
#                     prepare_transformer_decoder_params(*transformer_config[i])
#                 )
#             )
#         self.fc_out = Linear(transformer_config[-1][0], vocab_size)

#     def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None):
#         """
#         Args:
#             prev_output_tokens (LongTensor): previous decoder outputs of shape
#                 `(batch, tgt_len)`, for input feeding/teacher forcing
#             encoder_out (Tensor, optional): output from the encoder, used for
#                 encoder-side attention
#             incremental_state (dict): dictionary used for storing state during
#                 :ref:`Incremental decoding`
#         Returns:
#             tuple:
#                 - the last decoder layer's output of shape `(batch, tgt_len,
#                   vocab)`
#                 - the last decoder layer's attention weights of shape `(batch,
#                   tgt_len, src_len)`
#         """
#         target_padding_mask = (
#             (prev_output_tokens == self.padding_idx).to(prev_output_tokens.device)
#             if incremental_state is None
#             else None
#         )

#         if incremental_state is not None:
#             prev_output_tokens = prev_output_tokens[:, -1:]

#         # embed tokens
#         x = self.embed_tokens(prev_output_tokens)

#         # B x T x C -> T x B x C
#         x = self._transpose_if_training(x, incremental_state)

#         for layer in self.conv_layers:
#             if isinstance(layer, LinearizedConvolution):
#                 x = layer(x, incremental_state)
#             else:
#                 x = layer(x)

#         # B x T x C -> T x B x C
#         x = self._transpose_if_inference(x, incremental_state)

#         # decoder layers
#         for layer in self.layers:
#             if isinstance(layer, TransformerDecoderLayer):
#                 x, *_ = layer(
#                     x,
#                     (encoder_out["encoder_out"] if encoder_out is not None else None),
#                     (
#                         encoder_out["encoder_padding_mask"].t()
#                         if encoder_out["encoder_padding_mask"] is not None
#                         else None
#                     ),
#                     incremental_state,
#                     self_attn_mask=(
#                         self.buffered_future_mask(x)
#                         if incremental_state is None
#                         else None
#                     ),
#                     self_attn_padding_mask=(
#                         target_padding_mask if incremental_state is None else None
#                     ),
#                 )
#             else:
#                 x = layer(x)

#         # T x B x C -> B x T x C
#         x = x.transpose(0, 1)

#         x = self.fc_out(x)

#         return x, None

#     def buffered_future_mask(self, tensor):
#         dim = tensor.size(0)
#         if (
#             not hasattr(self, "_future_mask")
#             or self._future_mask is None
#             or self._future_mask.device != tensor.device
#         ):
#             self._future_mask = torch.triu(
#                 utils.fill_with_neg_inf(tensor.new(dim, dim)), 1
#             )
#         if self._future_mask.size(0) < dim:
#             self._future_mask = torch.triu(
#                 utils.fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1
#             )
#         return self._future_mask[:dim, :dim]

#     def _transpose_if_training(self, x, incremental_state):
#         if incremental_state is None:
#             x = x.transpose(0, 1)
#         return x

#     def _transpose_if_inference(self, x, incremental_state):
#         if incremental_state:
#             x = x.transpose(0, 1)
#         return x
