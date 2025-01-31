# OSLSP
Ordinal scale Similarity Proportion Learning. <br>
Similarity Proportion Loss for Weakly Supervised VIT using Ordinal scale Label 

# Quick start

```
python3 -m torch.distributed.launch \
--nproc_per_node=2 --master_port 8003 OSLSP/src/train.py \
--batch_size_per_gpu 1 \
--local_crops_number=1 \
--data_path ./data/PILOT_TEST \
--output_dir ./OSLSP_output
```
