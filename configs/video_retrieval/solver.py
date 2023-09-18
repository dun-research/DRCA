total_epochs = 100
warmup_epochs=5

base_lr = 1.5e-4
batch_size_per_gpu = 16
n_gpus = 8
batch_size = batch_size_per_gpu*n_gpus
lr = batch_size * base_lr / 256

optimizer = dict(
    type='AdamW',
    lr=lr,
    paramwise_cfg=dict(
       custom_keys={
           '.backbone.cls_token': dict(decay_mult=0.0),
           '.backbone.pos_embed': dict(decay_mult=0.0),
           '.backbone.time_embed': dict(decay_mult=0.0)
       }),
    weight_decay=1e-4,
    ) 

lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    min_lr_ratio=1e-6,
    warmup='linear',
    warmup_ratio=1e-3,
    warmup_iters=20,
    warmup_by_epoch=True)


optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))

