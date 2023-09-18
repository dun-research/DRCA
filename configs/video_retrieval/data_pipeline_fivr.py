img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_bgr=False)

# ======================================== path settings ========================================
data_root = 'data/kinetics400/rawframes_train'
ann_file_train = 'data/kinetics400/kinetics400_train_list_rawframes.txt'

fivr_version = "5k" 
if fivr_version == "5k":
    query_file="data/FIVR-200K/annotation/fivr-5k-queries.txt"
    database_file="data/FIVR-200K//annotation/fivr-5k-database.txt"
    anno_file="data/FIVR-200K/annotation/fivr-200k-visil.pkl"
    data_root="data/FIVR-200K/"
elif fivr_version == "200k":
    dataset_name = "fivr_200k"
    query_file="data/FIVR-200K/annotation/fivr-200k-queries.txt"
    database_file="data/FIVR-200K//annotation/fivr-200k-database.txt"
    anno_file="data/FIVR-200K//annotation/fivr-200k-visil.pkl"
    data_prefix="data/FIVR-200K/"
else:
    raise ValueError("fivr_version must be 5k or 200k")
# ======================================== path settings ========================================

train_pipeline = [
    dict(type='Flip', flip_ratio=0.5),
    dict(type='ImageAug', ),
    dict(type='PadAndResize', size=(224,224)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label', 'video_id', ], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label', 'video_id', ])
]


train_dataset = dict(
            type="RetrievalTrainRawframeDataset",
            ann_file=ann_file_train,
            data_prefix=data_root,
            clip_len=8, 
            scale_range=(256, 320),
            crop_size=224,
            long_frame_interval=32, 
            short_frame_interval=16,
            pipeline=train_pipeline)


fivr_pipeline = [
            dict(type="GeneralVideoDecoder", clip_len=8,  backend="pyav", fps_interval=1),
            dict(type='Resize', scale=(-1, 224)),
            dict(type='CenterCrop', crop_size=224),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label', ], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label', ]),

]
fivr_dataset = dict(
    type="FIVREvalDataset",
    query_file=query_file,
    database_file=database_file,
    anno_file=anno_file,
    data_prefix=data_root,
    pipeline=fivr_pipeline,
)

data = dict(
    videos_per_gpu=16,
    workers_per_gpu=8,
    test_dataloader=dict(videos_per_gpu=1),
    val_dataloader=dict(videos_per_gpu=1),
    train=train_dataset,
    val=fivr_dataset,
    test=fivr_dataset,
    )
evaluation = dict()
eval_config = dict()
