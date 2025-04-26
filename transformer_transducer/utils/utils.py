from typing import Any, Dict, List, Optional, Union
import torch
from pathlib import Path
import numpy as np


# def extract_fbank_features(
#     waveform: torch.FloatTensor,
#     sample_rate: int,
#     output_path: Optional[Path] = None,
#     n_mel_bins: int = 80,
#     overwrite: bool = False,
# ):
#     if output_path is not None and output_path.is_file() and not overwrite:
#         return

#     _waveform, _ = convert_waveform(waveform, sample_rate, to_mono=True)
#     # Kaldi compliance: 16-bit signed integers
#     _waveform = _waveform * (2 ** 15)
#     _waveform = _waveform.numpy()

#     features = _get_kaldi_fbank(_waveform, sample_rate, n_mel_bins)
#     if features is None:
#         features = _get_torchaudio_fbank(_waveform, sample_rate, n_mel_bins)
#     if features is None:
#         raise ImportError(
#             "Please install pyKaldi or torchaudio to enable fbank feature extraction"
#         )

#     if output_path is not None:
#         np.save(output_path.as_posix(), features)
#     return features
