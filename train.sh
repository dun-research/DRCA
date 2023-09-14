
python -m torch.distributed.launch --nproc_per_node=8 tools/train.py  configs/DRCA-S-K4-224.py \
	--cfg-options work_dir='work_dirs/DRCA/DRCA-S-K4-224'  


