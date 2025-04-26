#!/bin/bash
#SBATCH --job-name=vgg
#SBATCH -o /data/npl/Speech2Text/train_out/vgg_%j.out
#SBATCH --gres=gpu:a100:1
#SBATCH -N 1           # Số lượng node để chạy
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=50G



python /data/npl/Speech2Text/VietnameseASR-main/vggtransformer_asr/vggtransformer_train.py /data/npl/Speech2Text/VietnameseASR-main/vggtransformer_asr/hparams/vggtransformer.yaml