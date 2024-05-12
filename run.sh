# # first run this, do not run the second step
# python train_ext.py log.run=6 log.proj_name='SEED' data=SEED train.gpus=[0] train.valid_method='loo'

# second run this, before running config the checkpoint path, and don't run first step
python extract_fea.py log.run=6 log.proj_name='SEED' data=SEED train.gpus=[0] train.valid_method='loo' 

python train_mlp.py log.run=6 log.proj_name='SEED' data=SEED train.gpus=[0] train.valid_method='loo'