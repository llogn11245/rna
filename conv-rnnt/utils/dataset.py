import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch
import torchaudio
import torchaudio.transforms as T

# [{idx : {encoded_text : Tensor, wav_path : text} }]


def load_json(path):
    """
    Load a json file and return the content as a dictionary.
    """
    import json

    with open(path, "r", encoding='utf-8') as f:
        data = json.load(f)
    return data

class Vocab:
    def __init__(self, vocab_path):
        self.vocab = load_json(vocab_path)
        self.itos = {v: k for k, v in self.vocab.items()}
        self.stoi = self.vocab

    def get_sos_token(self):
        return self.stoi["<s>"]
    def get_eos_token(self):
        return self.stoi["</s>"]
    def get_pad_token(self):
        return self.stoi["<pad>"]
    def get_unk_token(self):
        return self.stoi["<unk>"]
    def __len__(self):
        return len(self.vocab)





class Speech2Text(Dataset):
    def __init__(self, json_path, vocab_path):
        super().__init__()
        self.data = load_json(json_path)
        self.vocab = Vocab(vocab_path)
        self.sos_token = self.vocab.get_sos_token()
        self.eos_token = self.vocab.get_eos_token()
        self.pad_token = self.vocab.get_pad_token()
        self.unk_token = self.vocab.get_unk_token()

        # self.freq_mask = T.FrequencyMasking(freq_mask_param=30)  
        # self.time_mask = T.TimeMasking(time_mask_param=40)
            
    def __len__(self):
        return len(self.data)
    
    def get_fbank(self, waveform, sample_rate=16000):
        # 1. Lấy STFT (not Mel)
        stft = torch.stft(
            input=waveform,
            n_fft=512,
            hop_length=int(0.010 * sample_rate),  # 10ms
            win_length=int(0.025 * sample_rate),  # 25ms
            window=torch.hamming_window(int(0.025 * sample_rate)),
            return_complex=True
        )

        # 2. Lấy magnitude và chỉ giữ 64 tần số đầu tiên (low frequency)
        mag = stft.abs()[:64, :]  # [F=64, T]

        # 3. Convert to log scale (avoid log(0))
        log_spec = torch.log10(mag + 1e-10)  # [64, T]

        # 4. Transpose về [T, 64]
        log_spec = log_spec.transpose(0, 1)  # [T, 64]

        # 5. Stack 3 frames with skip=3
        # e.g., frame t = [t, t+3, t+6] → dim = 64*3 = 192
        T = log_spec.shape[0]
        stacked = []
        for i in range(0, T - 6, 1):
            stacked.append(torch.cat([log_spec[i], log_spec[i+3], log_spec[i+6]], dim=-1))  # [192]
        stacked = torch.stack(stacked)  # [T', 192]

        # 6. Global mean and std normalization
        mean = stacked.mean(dim=0, keepdim=True)
        std = stacked.std(dim=0, keepdim=True)
        stacked = (stacked - mean) / (std + 1e-5)  # [T', 192]

        return stacked  # [T', 192]

    
    def extract_from_path(self, wave_path):
        waveform, sr = torchaudio.load(wave_path)
        waveform = waveform.squeeze(0)  # (channel,) -> (samples,)
        return self.get_fbank(waveform, sample_rate=sr)

    def __getitem__(self, idx):
        current_item = self.data[idx]
        wav_path = current_item["wav_path"]
        encoded_text = torch.tensor(current_item["encoded_text"] + [self.eos_token], dtype=torch.long)
        decoder_input = torch.tensor([self.sos_token] + current_item["encoded_text"], dtype=torch.long)
        fbank = self.extract_from_path(wav_path).float()  # [T, 80]
        
        return {
            "text": encoded_text,        # [T_text]
            "fbank": fbank,              # [T_audio, 80]
            "text_len": len(encoded_text),
            "fbank_len": fbank.shape[0],
            "decoder_input": decoder_input,  # [T_text + 1]
        }
    
from torch.nn.utils.rnn import pad_sequence

def calculate_mask(lengths, max_len):
    """Tạo mask cho các tensor có chiều dài khác nhau"""
    mask = torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]
    return mask

def speech_collate_fn(batch):
    decoder_outputs = [torch.tensor(item["decoder_input"]) for item in batch]
    texts = [item["text"] for item in batch]
    fbanks = [item["fbank"] for item in batch]
    text_lens = torch.tensor([item["text_len"] for item in batch], dtype=torch.long)
    fbank_lens = torch.tensor([item["fbank_len"] for item in batch], dtype=torch.long)

    padded_decoder_inputs = pad_sequence(decoder_outputs, batch_first=True, padding_value=0)
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)       # [B, T_text]
    padded_fbanks = pad_sequence(fbanks, batch_first=True, padding_value=0.0)   # [B, T_audio, 80]

    speech_mask=calculate_mask(fbank_lens, padded_fbanks.size(1))      # [B, T]
    text_mask=calculate_mask(text_lens, padded_texts.size(1))

    return {
        "decoder_input": padded_decoder_inputs,
        "text": padded_texts,
        "text_mask": text_mask,
        "text_len" : text_lens,
        "fbank_len" : fbank_lens,
        "fbank": padded_fbanks,
        "fbank_mask": speech_mask
    }