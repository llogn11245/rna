#!/bin/bash
#SBATCH --job-name=conformer
#SBATCH -o /data/npl/Speech2Text/train_out/conformer_%j.out
#SBATCH --gres=gpu:a100:1
#SBATCH -N 1           # Số lượng node để chạy
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=50G

python /data/npl/Speech2Text/VietnameseASR-main/transformer_asr/transformer_words_train.py /data/npl/Speech2Text/VietnameseASR-main/transformer_asr/hparams/conformer.yaml