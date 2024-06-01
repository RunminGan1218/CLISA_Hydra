run=4
dataset=SEED
gpus='[0]'
proj_name="$dataset""_msLen5_loo"
valid_method="loo"
logging=default
iftest=False

# you can run it in once
echo "proj: $proj_name run: $run gpus: $gpus valid_method: $valid_method"
# python train_ext.py log.run=$run log.proj_name=$proj_name data=$dataset \
#                     model.timeFilterLen=30 model.dilation_array=[1,3,6,12] \
#                     model.seg_att=15 model.avgPoolLen=15 model.timeSmootherLen=3 \
#                     train.gpus=$gpus train.valid_method=$valid_method \
#                     train.num_workers=8 \
#                     hydra/job_logging=$logging train.iftest=$iftest


# python extract_fea.py log.run=$run log.proj_name=$proj_name data=$dataset \
#                       model.timeFilterLen=30 model.dilation_array=[1,3,6,12] \
#                       model.seg_att=15 model.avgPoolLen=15 model.timeSmootherLen=3 \
#                       train.gpus=$gpus train.valid_method=$valid_method \
#                       train.num_workers=8 \
#                       hydra/job_logging=$logging train.iftest=$iftest

# export HYDRA_FULL_ERROR=1   
# python train_mlp.py log.run=$run log.proj_name=$proj_name data=$dataset \
#                     model.timeFilterLen=30 model.dilation_array=[1,3,6,12] \
#                     model.seg_att=15 model.avgPoolLen=15 model.timeSmootherLen=3 \
#                     train.gpus=$gpus train.valid_method=$valid_method \
#                     train.num_workers=8 \
#                     hydra/job_logging=$logging train.iftest=$iftest

python train_mlp.py log.run=$run log.proj_name=$proj_name data=$dataset \
                    model.timeFilterLen=30 model.dilation_array=[1,3,6,12] \
                    model.seg_att=15 model.avgPoolLen=15 model.timeSmootherLen=3 \
                    train.gpus=$gpus train.valid_method=$valid_method \
                    train.num_workers=8 \
                    hydra/job_logging=$logging train.iftest=$iftest \
                    mlp.wd=0.0022

# python train_mlp.py log.run=$run log.proj_name=$proj_name data=$dataset \
#                     model.timeFilterLen=30 model.dilation_array=[1,3,6,12] \
#                     model.seg_att=15 model.avgPoolLen=15 model.timeSmootherLen=3 \
#                     train.gpus=$gpus train.valid_method=$valid_method \
#                     train.num_workers=8 \
#                     hydra/job_logging=$logging train.iftest=$iftest \
#                     mlp.wd=0.005

# python train_mlp.py log.run=$run log.proj_name=$proj_name data=$dataset \
#                     model.timeFilterLen=30 model.dilation_array=[1,3,6,12] \
#                     model.seg_att=15 model.avgPoolLen=15 model.timeSmootherLen=3 \
#                     train.gpus=$gpus train.valid_method=$valid_method \
#                     train.num_workers=8 \
#                     hydra/job_logging=$logging train.iftest=$iftest \
#                     mlp.wd=0.011

# python train_mlp.py log.run=$run log.proj_name=$proj_name data=$dataset \
#                     model.timeFilterLen=30 model.dilation_array=[1,3,6,12] \
#                     model.seg_att=15 model.avgPoolLen=15 model.timeSmootherLen=3 \
#                     train.gpus=$gpus train.valid_method=$valid_method \
#                     train.num_workers=8 \
#                     hydra/job_logging=$logging train.iftest=$iftest \
#                     mlp.wd=0.025