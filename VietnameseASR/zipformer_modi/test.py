from hyperpyyaml import load_hyperpyyaml
from zipformer_model import ZipFormerEncoderDecoder  # ⚠️ import tất cả lớp dùng trong YAML

hparams_file = '/data/npl/Speech2Text/VietnameseASR-main/zipformer_file/hparams/zipformer.yaml'
with open(hparams_file) as f:
    hparams = load_hyperpyyaml(f)

print(hparams)