checkpoint_config = dict(interval=5)
optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    paramwise_cfg=dict(
        custom_keys=dict({
            '.backbone.cls_token': dict(decay_mult=0.0),
            '.backbone.pos_embed': dict(decay_mult=0.0),
            '.backbone.time_embed': dict(decay_mult=0.0)
        })),
    weight_decay=0.0001,
    nesterov=True)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
lr_config = dict(policy='step', step=[10, 20])
total_epochs = 30
