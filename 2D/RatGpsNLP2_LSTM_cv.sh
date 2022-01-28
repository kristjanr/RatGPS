#!/bin/bash

#SBATCH -J=rat3 
#SBATCH --partition=gpu  
#SBATCH --gres=gpu:tesla:1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=16:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=renata.siimon@gmail.com
#SBATCH -o /gpfs/space/home/renata24/neuro/scripts_out/log_file_00-1_out-%j.txt
#SBATCH -e /gpfs/space/home/renata24/neuro/scripts_out/log_file_00-1_err-%j.txt

#module load python/3.6.3/virtenv
#source activate baka38 
cd /gpfs/space/home/renata24/neuro

python RatGpsNLP2_LSTM_cv.py