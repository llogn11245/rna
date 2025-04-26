import torch
from utils.dataset import Speech2Text, speech_collate_fn
from models.model import TransformerTransducer

# ==== Load Dataset ====
train_dataset = Speech2Text(
    json_path="/home/anhkhoa/transformer_transducer/data/train.json",
    vocab_path="/home/anhkhoa/transformer_transducer/data/vocab.json"
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    collate_fn = speech_collate_fn
)

# ==== Kiểm tra 1 batch ====
batch = next(iter(train_loader))

# print("✅ Batch loaded!")
# print("Fbank shape       :", batch['fbank'].shape)       # [B, T, 80]
# print("Fbank lengths     :", batch['fbank_len'])          # [B]
# print("Text shape        :", batch['text'].shape)         # [B, U]
# print("Text lengths      :", batch['text_len'])           # [B]

# ==== Load model (giả sử bạn có config) ====
model = TransformerTransducer(
    in_features=80,
    n_classes=len(train_dataset.vocab),
    n_layers=4,
    n_dec_layers=2,
    d_model=256,
    ff_size=1024,
    h=4,
    joint_size=512,
    enc_left_size=2,
    enc_right_size=2,
    dec_left_size=1,
    dec_right_size=1,
    p_dropout=0.1
)

def calculate_mask(lengths, max_len):
    """Tạo mask cho các tensor có chiều dài khác nhau"""
    mask = torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]
    return mask
    

with torch.no_grad():
    output, fbank_len, text_len = model(
        speech=batch["fbank"],           # [B, T, 80]
        speech_mask=batch["fbank_mask"],         # [B, T]
        text=batch["text"],              # [B, U]
        text_mask=batch["text_mask"]            # [B, U]
    )
    
print("✅ Model output shape:", output.shape)  # [B, T, U, vocab_size]
