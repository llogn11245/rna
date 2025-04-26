import os
import sys
import torch
import torch.nn as nn
import random
import logging
from pathlib import Path
import speechbrain as sb
import json
import re
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main

import torch
import torch.nn as nn
from collections.abc import Iterable
from itertools import repeat
from vggtrans import VGGTransformerEncoderDecoder

word2index = {}
index2word = {}
def load_label_json(labels_path):
    with open(labels_path, encoding="utf-8") as label_file:
        labels = json.load(label_file)
        word2index = dict()
        index2word = dict()
    
        for index, char in enumerate(labels):
            word2index[char] = index
            index2word[index] = char
            
        return word2index, index2word

word2index, index2word = load_label_json('/data/npl/Speech2Text/VietnameseASR-main/LSVSC_100_vocab.json') # Vocab file
print(len(word2index))
sos_token, eos_token, pad_token = word2index['<s>'], word2index['</s>'], word2index['_'] # <s>, </s>, _

logger = logging.getLogger(__name__)


def compute_lengths(features):
    """
    Tính toán độ dài tương đối của từng mẫu trong batch.
    
    Args:
        features: Tensor có shape (batch_size, time_steps, feature_dim)

    Returns:
        Tensor (batch_size,) chứa độ dài của từng sample theo tỷ lệ so với mẫu dài nhất.
    """
    batch_size, max_time_steps, _ = features.shape  # (8, 2050, 80)
    
    # Lấy số frame thực tế cho từng sample
    real_lengths = torch.tensor([features[i].sum(dim=-1).nonzero().shape[0] for i in range(batch_size)], dtype=torch.float32)

    # Chuẩn hóa về khoảng [0, 1] bằng cách chia cho số frame dài nhất
    l = real_lengths / max_time_steps

    return l

def compute_tokens_lens(tokens):
    """
    Tính độ dài tương đối của tokens trong batch, chuẩn hóa theo max_seq_len.

    Args:
        tokens: Tensor có shape (batch_size, max_seq_len)

    Returns:
        Tensor (batch_size,) chứa tỷ lệ token thực tế so với mẫu dài nhất.
    """
    batch_size, max_seq_len = tokens.shape  # (batch_size, max_seq_len)

    # Đếm số lượng token thực sự (khác 0)
    real_token_lengths = torch.tensor([(tokens[i] != 0).sum().item() for i in range(batch_size)], dtype=torch.float32)

    # Chuẩn hóa về [0, 1] bằng cách chia cho số token dài nhất trong batch
    tokens_lens = real_token_lengths / max_seq_len

    return tokens_lens


class ASR(sb.core.Brain):
    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)

        feats, l = batch['features'] # MFCC
        tokens_bos, _ = batch.tokens_bos # <S> A B C D E F
        current_epoch = self.hparams.epoch_counter.current # Epoch hiện tại
        feats = self.modules.normalize(feats, l, epoch=current_epoch)
        


        if stage == sb.Stage.TRAIN: # augment
            if hasattr(self.hparams, "augmentation"):
                feats = self.hparams.augmentation(feats, l)

        # forward modules
        src = feats
        enc_out, pred = self.modules.model(
            src, tokens_bos, l, pad_idx=pad_token,
        ) # encoder and decoder outputs


        # ctc log-probabilities
        logits = self.modules.ctc_lin(enc_out)
        p_ctc = self.hparams.log_softmax(logits)

        # seq2seq log-probabilities
        pred = self.modules.seq_lin(pred)
        p_seq = self.hparams.log_softmax(pred)

        # outputs
        hyps = None
        if stage == sb.Stage.TRAIN:
            hyps = None
        elif stage == sb.Stage.VALID:
            hyps = None
            current_epoch = self.hparams.epoch_counter.current
            if current_epoch % self.hparams.valid_search_interval == 0:
                hyps = self.hparams.valid_search(enc_out.detach(), l)

        elif stage == sb.Stage.TEST:
            hyps = self.hparams.valid_search(enc_out.detach(), l)
        return p_ctc, p_seq, l, hyps

    def compute_objectives(self, predictions, batch, stage):
        (p_ctc, p_seq, wav_lens, hyps,) = predictions # compute_features outputs

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos # A B C D </s>
        tokens, tokens_lens = batch.encoded_text
        
        
        # raise
        loss_seq = self.hparams.seq_cost(
            p_seq, tokens_eos, length=tokens_eos_lens
        ).sum() # Tính loss kết quả của decoder
        
        loss_ctc = self.hparams.ctc_cost(
            p_ctc, tokens, wav_lens, tokens_lens
        ).sum() # Tính loss kết quả ctc

        loss = (
            self.hparams.ctc_weight * loss_ctc
            + (1 - self.hparams.ctc_weight) * loss_seq
        )
      
        if stage != sb.Stage.TRAIN:
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if current_epoch % valid_search_interval == 0 or (
                stage == sb.Stage.TEST
            ):
                target_words, target_lens = batch.encoded_text
                # CER
                self.cer_metric.append(ids, hyps[0], target_words,
                                           target_len= target_lens,
                                       ind2lab=lambda batch: [list(' '.join([index2word[int(x)] for x in seq])) for seq in batch])
                # WER
                self.wer_metric.append(ids, hyps[0], target_words,
                                           target_len= target_lens, ind2lab=lambda batch: [[index2word[int(x)] for x in seq] for seq in batch])

            self.acc_metric.append(p_seq, tokens_eos, tokens_eos_lens)
        return loss
    
    def on_evaluate_start(self, max_key=None, min_key=None):
        super().on_evaluate_start()

        ckpts = self.checkpointer.find_checkpoints(
            max_key=max_key, min_key=min_key
        )
        ckpt = sb.utils.checkpoints.average_checkpoints(
            ckpts, recoverable_name="model", device=self.device
        )

        self.hparams.model.load_state_dict(ckpt, strict=True)
        self.hparams.model.eval()

        print("Loaded the average")

    def evaluate_batch(self, batch, stage):
        with torch.no_grad():
            predictions = self.compute_forward(batch, stage=stage)
            loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        if stage != sb.Stage.TRAIN:
            self.acc_metric = self.hparams.acc_computer() # Tính độ chính xác
            self.cer_metric = self.hparams.error_rate_computer() # Tính CER
            self.wer_metric = self.hparams.error_rate_computer() # Tính WER
    
    def on_stage_end(self, stage, stage_loss, epoch):
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats # epoch training details
        else:
            stage_stats["ACC"] = self.acc_metric.summarize()
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if (
                current_epoch % valid_search_interval == 0
                or stage == sb.Stage.TEST
            ):
                stage_stats["CER"] = self.cer_metric.summarize("error_rate") # Tính trung bình CER
                stage_stats["WER"] = self.wer_metric.summarize("error_rate") # Tính trung bình WER

        # log stats and save checkpoint at end-of-epoch
        if stage == sb.Stage.VALID and sb.utils.distributed.if_main_process():
                
            lr = self.hparams.noam_annealing.current_lr
            steps = self.optimizer_step
            optimizer = self.optimizer.__class__.__name__

            epoch_stats = {
                "epoch": epoch,
                "lr": lr,
                "steps": steps,
                "optimizer": optimizer,
            }

            self.hparams.train_logger.log_stats(
                stats_meta=epoch_stats,
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )

            self.checkpointer.save_and_keep_only(
                meta={"ACC": stage_stats["ACC"], "epoch": epoch},
                max_keys=["ACC"],
                num_to_keep=5,
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.cer_file, "w") as w:
                self.cer_metric.write_stats(w) 
            with open(self.hparams.wer_file, "w") as w:
                self.wer_metric.write_stats(w) 

    def fit_batch(self, batch):
        should_step = self.step % self.grad_accumulation_factor == 0
        # Managing automatic mixed precision
        if self.auto_mix_prec:
            with torch.autocast(torch.device(self.device).type):
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)

            # Losses are excluded from mixed precision to avoid instabilities
            loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            with self.no_sync(not should_step):
                self.scaler.scale(
                    loss / self.grad_accumulation_factor
                ).backward()
            if should_step:
                self.scaler.unscale_(self.optimizer)
                if self.check_gradients(loss):
                    self.scaler.step(self.optimizer)
                self.scaler.update()
                self.zero_grad()
                self.optimizer_step += 1
                self.hparams.noam_annealing(self.optimizer)
        else:
            outputs = self.compute_forward(batch, sb.Stage.TRAIN)
            loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            with self.no_sync(not should_step):
                (loss / self.grad_accumulation_factor).backward()
            if should_step:
                if self.check_gradients(loss):
                    self.optimizer.step()
                self.zero_grad()
                self.optimizer_step += 1
                self.hparams.noam_annealing(self.optimizer) # Learning rate scheduler

        self.on_fit_batch_end(batch, outputs, loss, should_step)
        return loss.detach().cpu()

@sb.utils.data_pipeline.takes("wav", "text")
@sb.utils.data_pipeline.provides("features", "encoded_text", "tokens_bos", "tokens_eos", "feat_length", "target_length")
def parse_characters(path, text):
    sig  = sb.dataio.dataio.read_audio(f"{hparams['dataset_dir']}/{path}")
    features = hparams['compute_features'](sig.unsqueeze(0)).squeeze(0) # Đọc MFCC
    
    yield features
    
    # t = re.sub(r"[^\w\s]", ' ', text.lower()).strip()
    # t = re.sub(r"[\t\n\r\d]", ' ', t).replace(" ", " ")
    # t = re.sub(r"\s+", ' ', t)

    encoded_text = [word2index.get(x, word2index["unk"]) for x in text.split(' ')]
    # print(encoded_text)
    yield torch.LongTensor(encoded_text)
    yield torch.LongTensor([sos_token] + encoded_text)
    yield torch.LongTensor(encoded_text + [eos_token])

    yield features.size(0)

    yield len(encoded_text)+2

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_dataset(path, shuffle):
    data = load_json(path)

    if shuffle:
        keys = list(data.keys())
        random.shuffle(keys)
        shuffled_data = {}
        for key in keys:
            shuffled_data[key] = data[key]
        dataset = sb.dataio.dataset.DynamicItemDataset(shuffled_data)

    else:
        dataset = sb.dataio.dataset.DynamicItemDataset(data)

    dataset.add_dynamic_item(parse_characters)
    dataset.set_output_keys(['id','features', 'encoded_text', 'tokens_bos', 'tokens_eos', 'feat_length', 'target_length'])
    return dataset

# override
overrides = {
    'output_neurons': len(word2index),
    'bos_index': sos_token,
    'eos_index': eos_token,
    'pad_index': pad_token
}

def get_hparams():
    hparams_file, run_opts, temp_overrides = sb.parse_arguments(sys.argv[1:])

    if temp_overrides:
        temp_overrides = load_hyperpyyaml(temp_overrides)
        overrides.update(temp_overrides)

    with open(hparams_file) as f:
        hparams = load_hyperpyyaml(f, overrides)
    
    # Output directory
    sb.create_experiment_directory(
        experiment_directory = hparams["output_dir"],
        hyperparams_to_save = hparams_file,
        overrides=overrides
    )

    return hparams


from tqdm import tqdm
if __name__ == "__main__":

    hparams = get_hparams()

    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["Adam"],
        hparams=hparams,
        run_opts = {
            'device': 'cuda',
            'max_grad_norm': hparams['max_grad_norm'],
            'auto_mix_prec': True  # Thêm dòng này
        },
        checkpointer=hparams["checkpointer"],
    )

    train_dataloader_opts = hparams["train_dataloader_opts"]
    valid_dataloader_opts = hparams["valid_dataloader_opts"]

    # Datasets
    train_dataset = get_dataset(hparams['train_dataset'], False) if hparams['train_dataset'] else None
    valid_dataset = get_dataset(hparams['valid_dataset'], False) if hparams['valid_dataset'] else None
    test_dataset = get_dataset(hparams['test_dataset'], False) if hparams['test_dataset'] else None
    
    # Training + validation
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_dataset,
        valid_dataset,
        train_loader_kwargs=train_dataloader_opts,
        valid_loader_kwargs=valid_dataloader_opts,
    )

    # Testing

    asr_brain.hparams.cer_file = os.path.join(
        hparams["output_dir"], "cer_valid.txt"
    )
    asr_brain.hparams.wer_file = os.path.join(
        hparams["output_dir"], "wer_valid.txt"
    )
    
    asr_brain.evaluate(
        valid_dataset,
        max_key="ACC",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )


    asr_brain.hparams.cer_file = os.path.join(
        hparams["output_dir"], "cer_test.txt"
    )
    asr_brain.hparams.wer_file = os.path.join(
        hparams["output_dir"], "wer_test.txt"
    )

    asr_brain.evaluate(
        test_dataset,
        max_key="ACC",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )