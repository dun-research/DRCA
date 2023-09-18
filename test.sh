
# evaluate video classification
python -m torch.distributed.launch --nproc_per_node=8 tools/test.py \
     configs/video_classification/DRCA-S-K4-224.py  weights/DRCA-S-K4-224-minik.pth \
     -la pytorch --out outputs/DRCA-S-K4-224-minik


# evaluate video retrieval
python -m torch.distributed.launch --nproc_per_node=8 tools/test.py \
     configs/video_retrieval/DRCA-B-K4-224.py  weights/DRCA-B-K4-224-fivr.pth \
     -la pytorch --out outputs/DRCA-S-K4-224-fivr

