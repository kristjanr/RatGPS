#!/usr/bin/env bash
                                                                                                                                                            
label=$1
seq_len=100


for i in 1400; # use window size of 1400 only
do
at=$(($i/40)) # unsure what does this mean... yet
#for v in 1 2 3 4 5; # commented out. This was just to run the exact same train cycle for five times.
v=1
#do
  python3 ratcvfit.py "$label"_models/simple_window_scan_"$label"_1x"$i"_seq"$seq_len"_ep50_b64_"$v" --features data/"$label"_1x"$i"_at"$at"_step200_bin100-RAW_feat.dat --locations data/"$label"_1x"$i"_at"$at"_step200_bin100-RAW_pos.dat --batch_size 64 --epochs 50 --rnn lstm --layers 2 --hidden_nodes 512 --cvfolds 10 --save_best_model_only false --seqlen 100 --verbose 2
  sleep 3
#done
done
