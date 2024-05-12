python train_ext.py log.run=5 log.proj_name='SEED' data=SEED train.gpus=[0]
python extract_fea.py log.run=5 log.proj_name='SEED' data=SEED train.gpus=[0]
python train_mlp.py log.run=5 log.proj_name='SEED' data=SEED train.gpus=[0]