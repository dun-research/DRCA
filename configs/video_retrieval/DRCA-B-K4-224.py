_base_ = ['../_base_/default_runtime.py', './DRCA-B-224.py',
                  './data_pipeline_fivr.py', './solver.py']

model = dict(
    backbone=dict(
        comp_insert_layer=3,
        comp_k=4,
    ))

work_dir = 'work_dirs/DRCA/DRCA-B-K4-224-fivr'
