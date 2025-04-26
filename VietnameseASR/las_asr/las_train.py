import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
import sys
import json
import torch
import re
import os
import random

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

word2index, index2word = load_label_json('../LSVSC_100_vocab.json')
print(len(word2index))
sos_token, eos_token, pad_token = word2index['<s>'], word2index['</s>'], word2index['_']

class ASR(sb.core.Brain):
    def fuck_this(self):
        self.hparams.criterion = self.hparams.criterion.to(self.device)
    def compute_forward(self, batch, stage):
        teacher_forcing = 0

        batch = batch.to(self.device)
        
        feats, _ = batch.features
        feat_length = batch.feat_length
        scripts, _ = batch.encoded_text

        current_epoch = self.hparams.epoch_counter.current
        feats = self.hparams.normalize(feats, feat_length, epoch=current_epoch)

        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                feats = self.hparams.augmentation(feats)
            teacher_forcing = 1
        feats = feats.permute(0,2,1)
        feats = feats.unsqueeze(-3)

        logit = self.modules.seq2seq(feats, feat_length, scripts, teacher_forcing)
        return logit
    
    def compute_objectives(self, logit, batch, stage):
        ids = batch.id
        tokens_eos, l = batch.tokens_eos
        target = tokens_eos

        logit_s = torch.stack(logit, dim=1).to(self.device)
        y_hat = logit_s.max(-1)[1]

        if stage != sb.Stage.TRAIN:
            logit_s = logit_s[:, :target.size(1), :]

        loss = self.hparams.criterion(logit_s.contiguous().view(-1, logit_s.size(-1)), target.contiguous().view(-1))

        if stage != sb.Stage.TRAIN:
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if current_epoch % valid_search_interval == 0 or (
                stage == sb.Stage.TEST
            ):
                self.cer_metric.append(ids, y_hat, target,
                                           target_len= l,
                                       ind2lab=lambda batch: self.ind2char(batch))
                
                self.wer_metric.append(ids, y_hat, target,
                                        target_len= l,
                                    ind2lab = lambda batch: self.ind2word(batch))

            # compute the accuracy of the one-step-forward prediction
            self.acc_metric.append(logit_s, target, l)

        return loss
    
    def ind2char(self, batch):
        haha = []
        for seq in batch:
            temp = []
            for x in seq:
                w = int(x)
                if w == eos_token:
                    break
                temp.append(index2word[w])
            haha.append(list(' '.join(temp)))

        return haha

    def ind2word(self, batch):
        haha = []
        for seq in batch:
            temp = []
            for x in seq:
                w = int(x)
                if w == eos_token:
                    break
                temp.append(index2word[w])
            haha.append(temp)

        return haha

    def on_evaluate_start(self, max_key=None, min_key=None):
        """perform checkpoint averge if needed"""
        super().on_evaluate_start()

        ckpts = self.checkpointer.find_checkpoints(
            max_key=max_key, min_key=min_key
        )
        ckpt = sb.utils.checkpoints.average_checkpoints(
            ckpts, recoverable_name="model", device=self.device
        )

        self.hparams.model.load_state_dict(ckpt, strict=True)
        # self.hparams.normalize
        self.hparams.model.eval()
        print("Loaded the average")

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        with torch.no_grad():
            predictions = self.compute_forward(batch, stage=stage)
            loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()
    
    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.acc_metric = self.hparams.acc_computer()
            self.cer_metric = self.hparams.error_rate_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["ACC"] = self.acc_metric.summarize()
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if (
                current_epoch % valid_search_interval == 0
                or stage == sb.Stage.TEST
            ):
                stage_stats["CER"] = self.cer_metric.summarize("error_rate")
                stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # log stats and save checkpoint at end-of-epoch
        if stage == sb.Stage.VALID and sb.utils.distributed.if_main_process():

            lr = self.optimizer.param_groups[0]['lr']
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
        outputs = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
        loss.backward()
        
        if self.optimizer_step == 0:
            for g in self.optimizer.param_groups:
                g['lr'] = 0
                self.hparams.scheduler.get_next_value()

        # if self.check_gradients(loss):
        #     print('wtf')
        torch.nn.utils.clip_grad_norm_(self.modules.parameters(), self.hparams.max_grad_norm)
        self.optimizer.step()
        self.zero_grad()
        self.optimizer_step += 1

        next_lr = self.hparams.scheduler.get_next_value()
        for g in self.optimizer.param_groups:
            g['lr'] = next_lr

        self.on_fit_batch_end(batch, outputs, loss, True)
        return loss.detach().cpu()

@sb.utils.data_pipeline.takes("wav", "text")
@sb.utils.data_pipeline.provides("features", "encoded_text","tokens_eos", "feat_length")
def parse_characters(path, text):
    sig  = sb.dataio.dataio.read_audio(f"{hparams['dataset_dir']}/{path}")
    features = hparams['compute_features'](sig.unsqueeze(0)).squeeze(0)
    yield features
    
    t = re.sub(r"[^\w\s]", ' ', text.lower()).strip()
    t = re.sub(r"[\t\n\r\d]", ' ', t).replace(" ", " ")
    t = re.sub(r"\s+", ' ', t)

    encoded_text = [word2index.get(x) for x in t.split(' ')]

    yield torch.LongTensor([sos_token] + encoded_text + [eos_token])

    yield torch.LongTensor(encoded_text + [eos_token])

    yield features.size(0)

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
    dataset.set_output_keys(["id", "features", "encoded_text", "tokens_eos", "feat_length"])
    return dataset

overrides = {
    'vocab_size': len(word2index),
    'sos_token': sos_token,
    'eos_token': eos_token,
}

def get_hparams():
    hparams_file, run_opts, temp_overrides = sb.parse_arguments(sys.argv[1:])

    if temp_overrides:
        temp_overrides = load_hyperpyyaml(temp_overrides)
        overrides.update(temp_overrides)

    with open(hparams_file) as f:
        hparams = load_hyperpyyaml(f, overrides)
    
    sb.create_experiment_directory(
        experiment_directory = hparams["output_dir"],
        hyperparams_to_save = hparams_file,
        overrides=overrides
    )

    return hparams
from speechbrain.dataio.dataloader import SaveableDataLoader
from speechbrain.dataio.batch import PaddedBatch
if __name__ == '__main__':
    hparams = get_hparams()

    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["Adam"],
        hparams=hparams,
        run_opts = {
            'device': 'cuda',
            'max_grad_norm': hparams['max_grad_norm']
        },
        checkpointer=hparams["checkpointer"],
    )
    asr_brain.fuck_this()

    train_dataloader_opts = hparams["train_dataloader_opts"]
    valid_dataloader_opts = hparams["valid_dataloader_opts"]

    train_dataset = get_dataset(hparams['train_dataset'], True) if hparams['train_dataset'] else None
    valid_dataset = get_dataset(hparams['valid_dataset'], False) if hparams['valid_dataset'] else None
    test_dataset = get_dataset(hparams['test_dataset'], False) if hparams['test_dataset'] else None


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