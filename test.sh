python -m torch.distributed.launch --master_port 55 --nproc_per_node=8 tools/test.py \
     configs/DRCA-S-K2-224.py  weights/T2/DRCA-S-K4-224-minik.pth -la pytorch --out outputs/DRCA-S-K2-224
