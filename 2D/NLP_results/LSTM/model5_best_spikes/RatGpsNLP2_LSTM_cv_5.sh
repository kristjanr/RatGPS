#!/bin/bash

#SBATCH -J=rat5 
#SBATCH --partition=gpu  # gres=gpu:tesla:1
#SBATCH --ntasks=64
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=renata.siimon@gmail.com
#SBATCH -o /gpfs/space/home/renata24/neuro/scripts_out/log_file_00-1_out-%j.txt
#SBATCH -e /gpfs/space/home/renata24/neuro/scripts_out/log_file_00-1_err-%j.txt

#module load python/3.6.3/virtenv
#source activate baka38 
cd /gpfs/space/home/renata24/neuro

python RatGpsNLP2_LSTM_cv_5.py