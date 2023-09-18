_base_ = ['../_base_/default_runtime.py', './DRCA-S-224.py',
                  './data_pipeline_fivr.py', './solver.py']

model = dict(
    backbone=dict(
        comp_insert_layer=3,
        comp_k=3,
    ))

work_dir = 'work_dirs/DRCA/DRCA-S-K3-224-fivr'
