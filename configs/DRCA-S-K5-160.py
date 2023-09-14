_base_ = ['./DRCA-S-224.py']

model = dict(
    backbone=dict(
        img_size=160,
        comp_insert_layer=3,
        comp_k=5,
    ))
