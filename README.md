# OSLSP
Ordinal scale Similarity Proportion Learning. <br>
Similarity Proportion Loss for Weakly Supervised VIT using Ordinal scale Label 

# Quick start

```
cd /a/chan/dino
python3 -m torch.distributed.launch \
--nproc_per_node=2 --master_port 8003 OSLLP_main_dino.py \
--arch vit_base --patch_size=8 \
--saveckp_freq=1 \
--out_dim=768 \
--batch_size_per_gpu 1 \
--local_crops_number=1 \
--osllp_bins 31 \
--bag_num 100 \
--data_path /a/yu-yamaoka/Scientific_reports/DINO_INPUT/RandomCrip/0511TRAIN_Crop=64 \
--output_dir /a/yu-yamaoka/Scientific_reports/DINO_Output/OSLLP-KL_0326_SIZE=64_CenterTemp_NoneHEAD
```
