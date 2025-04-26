from vggblock import VGGBlock
import torch.nn as nn
import torch
from speechbrain.lobes.models.transformer.TransformerASR import (
    get_key_padding_mask, get_lookahead_mask, 
    length_to_mask
)
from speechbrain.lobes.models.transformer.Transformer import (
    TransformerEncoder, TransformerDecoder, NormalizedEmbedding,get_lookahead_mask
)
from speechbrain.nnet.containers import ModuleList
from speechbrain.nnet.linear import Linear
from fairseq.modules import (
    LinearizedConvolution
)
import math

def make_transformer_src_tgt_masks(
    src,
    tgt=None,
    wav_len=None,
    pad_idx=0,
    causal: bool = False,
    dynchunktrain_config = None,
):
    """This function generates masks for training the transformer model,
    opinionated for an ASR context with encoding masks and, optionally, decoding
    masks (if specifying `tgt`).

    Arguments
    ---------
    src : torch.Tensor
        The sequence to the encoder (required).
    tgt : torch.Tensor
        The sequence to the decoder.
    wav_len : torch.Tensor
        The lengths of the inputs.
    pad_idx : int
        The index for <pad> token (default=0).
    causal: bool
        Whether strict causality shall be used. See `make_asr_src_mask`
    dynchunktrain_config: DynChunkTrainConfig, optional
        Dynamic Chunk Training configuration. See `make_asr_src_mask`

    Returns
    -------
    src_key_padding_mask : torch.Tensor
        Key padding mask for ignoring padding
    tgt_key_padding_mask : torch.Tensor
        Key padding mask for ignoring padding
    src_mask : torch.Tensor
        Mask for ignoring invalid (e.g. future) timesteps
    tgt_mask : torch.Tensor
        Mask for ignoring invalid (e.g. future) timesteps
    """
    src_key_padding_mask = None

    # mask out audio beyond the length of audio for each batch
    if wav_len is not None:
        abs_len = torch.round(wav_len * src.shape[1])
        src_key_padding_mask = ~length_to_mask(abs_len).bool()

    # mask out the source
    src_mask = make_transformer_src_mask(
        src, causal=causal, dynchunktrain_config=dynchunktrain_config
    )

    # If no decoder in the transformer...
    if tgt is not None:
        tgt_key_padding_mask = get_key_padding_mask(tgt, pad_idx=pad_idx)
        tgt_mask = get_lookahead_mask(tgt)
    else:
        tgt_key_padding_mask = None
        tgt_mask = None

    return src_key_padding_mask, tgt_key_padding_mask, src_mask, tgt_mask

def make_transformer_src_mask(
    src: torch.Tensor,
    causal: bool = False,
    dynchunktrain_config= None,
) :
    """Prepare the source transformer mask that restricts which frames can
    attend to which frames depending on causal or other simple restricted
    attention methods.

    Arguments
    ---------
    src: torch.Tensor
        The source tensor to build a mask from. The contents of the tensor are
        not actually used currently; only its shape and other metadata (e.g.
        device).
    causal: bool
        Whether strict causality shall be used. Frames will not be able to
        attend to any future frame.
    dynchunktrain_config: DynChunkTrainConfig, optional
        Dynamic Chunk Training configuration. This implements a simple form of
        chunkwise attention. Incompatible with `causal`.

    Returns
    -------
    torch.Tensor
        A boolean mask Tensor of shape (timesteps, timesteps).
    """
    if causal:
        assert dynchunktrain_config is None
        return get_lookahead_mask(src)

    if dynchunktrain_config is None:
        return

    # The following is not really the sole source used to implement this,
    # but it helps introduce the concept.
    # ref: Unified Streaming and Non-streaming Two-pass End-to-end Model for Speech Recognition
    # https://arxiv.org/pdf/2012.05481.pdf
    timesteps = src.size(1)

    # Mask the future at the right of each chunk
    chunk_size = dynchunktrain_config.chunk_size
    num_chunks = timesteps // chunk_size
    timestep_idx = torch.arange(timesteps, device=src.device)
    mask_idx = torch.arange(
        chunk_size, chunk_size * (num_chunks + 2), chunk_size, device=src.device
    ).repeat_interleave(chunk_size)[:timesteps]
    src_mask = timestep_idx[None] >= mask_idx[:, None]

    # Mask the past at the left of each chunk (accounting for left context)
    # only relevant if using left context
    if not dynchunktrain_config.is_infinite_left_context():
        num_left_chunks = dynchunktrain_config.left_context_size
        mask_idx -= chunk_size * (num_left_chunks + 1)
        src_mask += timestep_idx[None] < mask_idx[:, None]

    return src_mask


def compute_wav_len_from_src(src, threshold=0.1):
    """Tính toán wav_len dựa trên src bằng cách đếm số frame có năng lượng vượt ngưỡng."""
    batch_size, max_length, _, _ = src.shape  # (B, T, C, Feat)

    # Tính tổng năng lượng trên mỗi frame (B, T)
    energy = src.abs().sum(dim=(2, 3))  # Tổng qua C và Feat → chỉ còn (B, T)

    # Đếm số frame có năng lượng lớn hơn threshold
    actual_length = (energy > threshold).sum(dim=1).float()  # (B,)

    # Chuẩn hóa về khoảng [0, 1]
    wav_len = actual_length / max_length
    return wav_len

def LinearizedConv1d(in_channels, out_channels, kernel_size, dropout=0, **kwargs):
    """Weight-normalized Conv1d layer optimized for decoding"""
    m = LinearizedConvolution(in_channels, out_channels, kernel_size, **kwargs)
    std = math.sqrt((4 * (1.0 - dropout)) / (m.kernel_size[0] * in_channels))
    nn.init.normal_(m.weight, mean=0, std=std)
    nn.init.constant_(m.bias, 0)
    return nn.utils.weight_norm(m, dim=2)

class VGGTransformerEncoderDecoder(nn.Module):
    def __init__(
        self, 
        input_feat_per_channel, 
        vggblock_config_enc, 
        vggblock_config_dec, 
        in_channels=1, 
        input_size=1280, 
        d_model=512, 
        dropout=0.1, 
        nhead=8, 
        num_encoder_layers=6, 
        num_decoder_layers=6,
        d_ffn=2048, 
        activation="relu", 
        normalize_before=True, 
        attention_type="self-attention", 
        causal=False, 
        kdim=None, 
        vdim=None, 
        layerdrop_prob=0.1, 
        output_hidden_states=False,
        tgt_vocab=1000,
        decoder_kdim=None,
        decoder_vdim=None
    ):
        super().__init__()

        self.conv_layers_dec_enc = nn.ModuleList()
        self.conv_layers_dec_dec = nn.ModuleList()
        self.in_channels = in_channels
        self.input_dim = input_feat_per_channel
        self.pooling_kernel_sizes = []

        # ====== Encoder VGG Blocks ======
        for _, config in enumerate(vggblock_config_enc):
            out_channels, conv_kernel_size, pooling_kernel_size, num_conv_layers, layer_norm = config # (32, 3, 1, 1, True)
            self.conv_layers_dec_enc.append(
                VGGBlock(
                    in_channels, out_channels, conv_kernel_size, pooling_kernel_size, 
                    num_conv_layers, input_dim=input_feat_per_channel, layer_norm=layer_norm
                )
            )
            in_channels = out_channels
            input_feat_per_channel = self.conv_layers_dec_enc[-1].output_dim

        # ====== Transformer Encoder ======
        self.custom_src_module = ModuleList(
            Linear(input_size=input_size, n_neurons=d_model, bias=True, combine_dims=False),
            torch.nn.Dropout(dropout),
        )

        self.encoder = TransformerEncoder(
            nhead=nhead, num_layers=num_encoder_layers, d_ffn=d_ffn, d_model=d_model,
            dropout=dropout, activation=activation, normalize_before=normalize_before,
            causal=causal, attention_type=attention_type, kdim=kdim, vdim=vdim,
            layerdrop_prob=layerdrop_prob,
        )


        # ====== Decoder VGG Blocks ======
        self.conv_layers_dec = nn.ModuleList()
        for i in range(len(vggblock_config_dec)):
            out_channels, kernel_size, layer_norm = vggblock_config_dec[i]
            if i == 0:
                conv_layer = LinearizedConv1d(
                    d_model, out_channels, kernel_size, padding=kernel_size - 1
                )
            else:
                conv_layer = LinearizedConv1d(
                    vggblock_config_dec[i - 1][0],
                    out_channels,
                    kernel_size,
                    padding=kernel_size - 1,
                )
            self.conv_layers_dec.append(conv_layer)
            if layer_norm:
                self.conv_layers_dec.append(nn.LayerNorm(out_channels))
            self.conv_layers_dec.append(nn.ReLU())

        # ====== Transformer Decoder ======
        self.custom_tgt_module = ModuleList(NormalizedEmbedding(d_model, tgt_vocab))

        self.decoder = TransformerDecoder(
            num_layers=num_decoder_layers, nhead=nhead, d_ffn=d_ffn, d_model=d_model,
            dropout=dropout, activation=activation, normalize_before=normalize_before,
            causal=True, attention_type="regularMHA", kdim=decoder_kdim, vdim=decoder_vdim,
        )

    def forward(self, src, tgt, wav_len, pad_idx=0):

        # Nếu src có 4 chiều (B, T, C, F), cần hoán vị lại
        # bsz, max_seq_len, _ = src.size()
        # src = src.view(bsz, max_seq_len, self.in_channels, self.input_dim)
        # src = src.transpose(1, 2).contiguous() # B C T F
        src = src.unsqueeze(2)
        src = src.transpose(1, 2).contiguous()
        
        for layer_idx, layer in enumerate(self.conv_layers_dec_enc):
            src = layer(src)


        src = src.permute(0, 2, 1, 3).contiguous()


        bsz, time, c, feat = src.shape
        src = src.view(bsz, time, c * feat)


        abs_len = torch.clamp(torch.round(wav_len * time).long(), max=time)


        src_key_padding_mask, tgt_key_padding_mask, src_mask, tgt_mask = make_transformer_src_tgt_masks(
            src, tgt, wav_len, causal=False, pad_idx=pad_idx
        )

        src = self.custom_src_module(src)


        encoder_out, _ = self.encoder(
            src=src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask, pos_embs=None
        )



        tgt = self.custom_tgt_module(tgt)
        tgt = tgt.transpose(0,1) # b t c -> t b c 


        for layer in self.conv_layers_dec:
            tgt = layer(tgt)
        
        tgt = tgt.transpose(0,1)
        # t b c -> btc
        decoder_out, _, _ = self.decoder(
            tgt=tgt, memory=encoder_out, memory_mask=None, tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=src_key_padding_mask
        )

        # print(f"🔹 [STEP 10] Transformer Decoder output.shape: {decoder_out.shape}")

        return encoder_out, decoder_out

    @torch.no_grad()
    def decode(self, tgt, encoder_out, enc_len=None):
        """This method implements a decoding step for the transformer model.

        Arguments
        ---------
        tgt : torch.Tensor
            The sequence to the decoder.
        encoder_out : torch.Tensor
            Hidden output of the encoder.
        enc_len : torch.LongTensor
            The actual length of encoder states.

        Returns
        -------
        prediction
        """
        tgt_mask = get_lookahead_mask(tgt)
        src_key_padding_mask = None
        if enc_len is not None:
            src_key_padding_mask = (1 - length_to_mask(enc_len)).bool()
        
        tgt = self.custom_tgt_module(tgt)

        tgt = tgt.transpose(0,1) # b t c -> t b c 


        for layer in self.conv_layers_dec:
            tgt = layer(tgt)
        
        tgt = tgt.transpose(0,1)

        pos_embs_target = None
        pos_embs_encoder = None

        prediction, self_attns, multihead_attns = self.decoder(
            tgt,
            encoder_out,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=src_key_padding_mask,
            pos_embs_tgt=pos_embs_target,
            pos_embs_src=pos_embs_encoder,
        )
        return prediction, multihead_attns[-1]

    def encode(
        self,
        src,
        wav_len=None,
        pad_idx=0,
        dynchunktrain_config = None,
    ):
        """
        Encoder forward pass

        Arguments
        ---------
        src : torch.Tensor
            The sequence to the encoder.
        wav_len : torch.Tensor, optional
            Torch Tensor of shape (batch, ) containing the relative length to padded length for each example.
        pad_idx : int
            The index used for padding.
        dynchunktrain_config : DynChunkTrainConfig
            Dynamic chunking config.

        Returns
        -------
        encoder_out : torch.Tensor
        """
        # reshape the src vector to [Batch, Time, Fea] if a 4d vector is given
        # bsz, max_seq_len, _ = src.size()
        # src = src.view(bsz, max_seq_len, self.in_channels, self.input_dim)
        # src = src.transpose(1, 2).contiguous() # B C T F
        
        src = src.unsqueeze(2)
        src = src.transpose(1, 2).contiguous()

        for layer_idx, layer in enumerate(self.conv_layers_dec_enc):
            src = layer(src)

        src = src.permute(0, 2, 1, 3).contiguous()

        bsz, time, c, feat = src.shape
        src = src.view(bsz, time, c * feat)

        abs_len = torch.clamp(torch.round(wav_len * time).long(), max=time)

        src_key_padding_mask, tgt_key_padding_mask, src_mask, tgt_mask = make_transformer_src_tgt_masks(
            src, None, wav_len, causal=False, pad_idx=pad_idx
        )

        src = self.custom_src_module(src)

        encoder_out, _ = self.encoder(
            src=src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask, pos_embs=None
        )
        return encoder_out