_base_ = ['../_base_/default_runtime.py', './DRCA-S-224.py',
                  './data_pipeline_minik.py', './solver.py']

model = dict(
    backbone=dict(
        comp_insert_layer=3,
        comp_k=4,
    ))

load_from =  'weights/DRCA-S-K8-full-224-minik.pth'
work_dir = 'work_dirs/video_classification/DRCA-S-K4-224'
