#!/bin/bash
#SBATCH --job-name=zipformer
#SBATCH -o /data/npl/Speech2Text/train_out/zip_%j.out
#SBATCH --gres=gpu:a100:1
#SBATCH -N 1           # Số lượng node để chạy
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=70G



python /data/npl/Speech2Text/VietnameseASR-main/zipformer_file/zipformer_train.py /data/npl/Speech2Text/VietnameseASR-main/zipformer_file/hparams/zipformer.yaml