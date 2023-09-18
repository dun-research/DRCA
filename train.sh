
# train video classification
python -m torch.distributed.launch --nproc_per_node=8 tools/train.py \
	configs/video_classification/DRCA-S-K4-224.py 


