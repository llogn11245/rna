import os
import sys
import torch
import torch.nn as nn
import math
# # === Add project paths ===
# project_paths = [
#     "/data/npl/Speech2Text/icefall/egs/librispeech/ASR/pruned_transducer_stateless7",
#     "/data/npl/Speech2Text/icefall",
#     "/data/npl/Speech2Text/icefall/icefall"
# ]
# for path in project_paths:
#     if path not in sys.path:
#         sys.path.append(path)

# === Import Zipformer ===
from zipformer import Zipformer  # Ensure this exists and is correct
from speechbrain.lobes.models.transformer.Transformer import (
    TransformerEncoder, TransformerDecoder, NormalizedEmbedding, get_lookahead_mask
)

from speechbrain.nnet.containers import ModuleList

from speechbrain.lobes.models.transformer.TransformerASR import (
    get_key_padding_mask, get_lookahead_mask, 
    length_to_mask
)



class PositionalEncoding(nn.Module):
    """This class implements the absolute sinusoidal positional encoding function.
    PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
    PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))
    Arguments
    ---------
    input_size: int
        Embedding dimension.
    max_len : int, optional
        Max length of the input sequences (default 2500).
    Example
    -------
    >>> a = torch.rand((8, 120, 512))
    >>> enc = PositionalEncoding(input_size=a.shape[-1])
    >>> b = enc(a)
    >>> b.shape
    torch.Size([1, 120, 512])
    """

    def __init__(self, input_size, max_len=2500):
        super().__init__()
        self.max_len = max_len
        pe = torch.zeros(self.max_len, input_size, requires_grad=False)
        positions = torch.arange(0, self.max_len).unsqueeze(1).float()
        denominator = torch.exp(
            torch.arange(0, input_size, 2).float()
            * -(math.log(10000.0) / input_size)
        )

        pe[:, 0::2] = torch.sin(positions * denominator)
        pe[:, 1::2] = torch.cos(positions * denominator)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Arguments
        ---------
        x : tensor
            Input feature shape (batch, time, fea)
        """
        return self.pe[:, : x.size(1)].clone().detach()


# === Utility function ===
def compute_x_lens_from_src(src: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    """
    Tính số lượng frame có năng lượng lớn hơn ngưỡng.
    
    Args:
        src: Tensor đầu vào có shape (B, T, F)
        threshold: Ngưỡng năng lượng để tính frame có tín hiệu

    Returns:
        Tensor (B,) chứa số frame thực tế (kiểu long)
    """
    # Tính năng lượng mỗi frame: abs và sum theo chiều feature
    energy = src.abs().sum(dim=-1)  # (B, T)

    # Đếm số frame có năng lượng > threshold
    lengths = (energy > threshold).sum(dim=1)  # (B,)

    return lengths

# === Encoder-Decoder wrapper ===
from typing import Tuple

def make_transformer_src_tgt_masks(
    src,
    tgt=None,
    wav_len=None,
    pad_idx=0,
    causal: bool = False,
    dynchunktrain_config = None,
    memory_len: int = None
):
    src_key_padding_mask = None

    if wav_len is not None and memory_len is not None:
        abs_len = torch.round(wav_len * memory_len)  # ✅ fix tại đây
        src_key_padding_mask = ~length_to_mask(abs_len).bool()

    src_mask = make_transformer_src_mask(
        src, causal=causal, dynchunktrain_config=dynchunktrain_config
    )

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

def compute_lengths(src: torch.Tensor, threshold: float = 0.0):
    energy = src.abs().sum(dim=-1)  # (B, T)
    x_lens = (energy > threshold).sum(dim=1)  # (B,)
    wav_len = x_lens.float() / src.shape[1]
    return x_lens, wav_len


class ZipFormerEncoderDecoder(nn.Module):
    def __init__(
        self,
        input_feat_per_channel: int = 80,
        tgt_vocab: int = 1000,
        output_downsampling_factor: int = 2,
        encoder_dims: Tuple[int] = (384, 384),
        attention_dim: Tuple[int] = (256, 256),
        encoder_unmasked_dims: Tuple[int] = (256, 256),
        zipformer_downsampling_factors: Tuple[int] = (2, 4),
        nhead: Tuple[int] = (8, 8),
        feedforward_dim: Tuple[int] = (1536, 2048),
        num_encoder_layers: Tuple[int] = (12, 12),
        dropout: float = 0.1,
        cnn_module_kernels: Tuple[int] = (31, 31),
        pos_dim: int = 4,
        warmup_batches: float = 4000.0,
        d_model: int = 384,
        causal = False,
        normalize_before: bool = False,
        activation = torch.nn.GELU,
        num_decoder_layers = 6,
        d_ffn = 2048,
        decoder_kdim = None,
        decoder_vdim = None,
        max_length = 5000,
        positional_encoding = "fixed_abs_sine"
    ) -> None:
        super().__init__()
        self.encoder = Zipformer(
            num_features=input_feat_per_channel,
            output_downsampling_factor=output_downsampling_factor,
            encoder_dims=encoder_dims,
            attention_dim=attention_dim,
            encoder_unmasked_dims=encoder_unmasked_dims,
            zipformer_downsampling_factors=zipformer_downsampling_factors,
            nhead=nhead,
            feedforward_dim=feedforward_dim,
            num_encoder_layers=num_encoder_layers,
            dropout=dropout,
            cnn_module_kernels=cnn_module_kernels,
            pos_dim=pos_dim,
            warmup_batches=warmup_batches
        )

        self.custom_tgt_module = ModuleList(
            NormalizedEmbedding(d_model, tgt_vocab)
        )
        
        self.causal = causal

        self.attention_type = "regularMHA"
        
        if positional_encoding == "fixed_abs_sine":
            self.positional_encoding = PositionalEncoding(d_model, max_length)
        self.decoder_kdim = decoder_kdim
        self.decoder_vdim = decoder_vdim
        self.decoder = TransformerDecoder(
                num_layers=num_decoder_layers,
                nhead=nhead[0],
                d_ffn=d_ffn,
                d_model=d_model,
                dropout=dropout,
                activation=activation,
                normalize_before=normalize_before,
                causal=True,
                attention_type="regularMHA",  # always use regular attention in decoder
                kdim=self.decoder_kdim,
                vdim=self.decoder_vdim,
            )
    def forward(self, src, tgt=None, wav_len=None, pad_idx=0):
        
        x_lens, wav_len = compute_lengths(src)
        
        encoder_out, _ = self.encoder(src, x_lens)
        memory_len = encoder_out.shape[1]
        

        (
            src_key_padding_mask,
            tgt_key_padding_mask,
            src_mask,
            tgt_mask,
        ) = make_transformer_src_tgt_masks(
            src, tgt, wav_len, causal=self.causal, pad_idx=pad_idx, memory_len = memory_len
        )


        tgt = self.custom_tgt_module(tgt)

        tgt = tgt + self.positional_encoding(tgt)
        pos_embs_target = None
        pos_embs_encoder = None

        decoder_out, _, _ = self.decoder(
            tgt=tgt,
            memory=encoder_out,
            memory_mask=None,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
            pos_embs_tgt=pos_embs_target,
            pos_embs_src=pos_embs_encoder,
        )

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
        """


        tgt_mask = get_lookahead_mask(tgt)
        src_key_padding_mask = None
        if enc_len is not None:
            src_key_padding_mask = (1 - length_to_mask(enc_len)).bool()

        tgt = self.custom_tgt_module(tgt)
        tgt = tgt + self.positional_encoding(tgt)
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
        x_lens, wav_len = compute_lengths(src)
        
        encoder_out, _ = self.encoder(src, x_lens)
        return encoder_out





if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running test on device: {device}")

    batch_size = 4
    time_steps = 2050
    features = 80
    vocab_size = 1000

    # (B, T, F) - input
    src = torch.randn(batch_size, time_steps, features).to(device)

    # (B, L) - target tokens
    tgt = torch.randint(0, vocab_size, (batch_size, 50)).to(device)

    # Instantiate model
    model = ZipFormerEncoderDecoder(input_feat_per_channel=features, tgt_vocab=vocab_size).to(device)
    model.eval()

    # Forward pass (encode + decode)
    try:
        out = model(src, tgt)
        print("Output (forward) encoder shape:", out[0].shape)
        print("Output (forward) decoder shape:", out[1].shape)
    except Exception as e:
        print("Error during forward pass:", e)

    print("\n=== Testing encode() and decode() ===")

    # Encode only
    try:
        encoded_out = model.encode(src)
        print("Encoded output shape:", encoded_out.shape)

        # Giả lập enc_len từ encode() bằng số bước thời gian
        enc_len = torch.full((batch_size,), encoded_out.shape[1], dtype=torch.long, device=device)

        # Decode
        decoded_out, final_attn = model.decode(tgt, encoded_out, enc_len=enc_len)
        print("Decoded output shape:", decoded_out.shape)
        print("Final attention shape:", final_attn.shape)
    except Exception as e:
        print("Error during encode/decode:", e)
