run=11
dataset=FACED
gpus='[0]'
proj_name="$dataset""_test"
valid_method=10
logging=default
iftest=True

# you can run it in once
echo "proj: $proj_name run: $run gpus: $gpus valid_method: $valid_method"
python train_ext.py log.run=$run log.proj_name=$proj_name data=$dataset \
                    train.gpus=$gpus train.valid_method=$valid_method \
                    hydra/job_logging=$logging train.iftest=$iftest


python extract_fea.py log.run=$run log.proj_name=$proj_name data=$dataset \
                      train.gpus=$gpus train.valid_method=$valid_method \
                      hydra/job_logging=$logging train.iftest=$iftest

python train_mlp.py log.run=$run log.proj_name=$proj_name data=$dataset \
                    train.gpus=$gpus train.valid_method=$valid_method \
                    hydra/job_logging=$logging train.iftest=$iftest