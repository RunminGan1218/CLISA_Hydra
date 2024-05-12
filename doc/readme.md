config the parameter and proj/exp in the cfgs/config.yaml
    change dataset in defaults
    change run in log   # caution
    change gpus in train
    change valid_method in run
    change pretrain timeLen timeStep class in data


run python file
1. run train_ext.py  pretrain 
    config the cp_dir 
2. run extract_fea.py   running norm lds save
3. run train_mlp.py   finetune

caution:
To ensure code security, please ensure that the same run_no is used when running three code files within a trial.


use wandb to see results

remain to be optimized:
DataModule:
    integrate 3 datamodule to one(only different in load_data)
    use cfg.data as arguments
    <!-- integrate pretrain(ext) and finetune(mlp) -->


