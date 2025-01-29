# OSLSP
Ordinal scale Similarity Proportion Learning. <br>
Similarity Proportion Loss for Weakly Supervised VIT using Ordinal scale Label 

# Quick start

```
python3 -m torch.distributed.launch \
--nproc_per_node=2 --master_port 8003 OSLSP/src/train.py \
--arch vit_base --patch_size=8 \
--saveckp_freq=1 \
--out_dim=768 \
--batch_size_per_gpu 1 \
--local_crops_number=1 \
--osllp_bins 31 \
--bag_num 100 \
--data_path data/PILOT_TEST/output \
--output_dir data/PILOT_TEST
```
