run=2
dataset=FACED
gpus='[0]'
proj_name="$dataset""_test_loo"
valid_method=10
iftest=True
logging=default

export HYDRA_FULL_ERROR=1
python interp_mlp.py log.run=$run log.proj_name=$proj_name data=$dataset \
                    model.timeFilterLen=30 model.dilation_array=[1,3,6,12] \
                    model.seg_att=15 model.avgPoolLen=15 model.timeSmootherLen=3 \
                    train.gpus=$gpus train.valid_method=$valid_method \
                    train.num_workers=8 \
                    hydra/job_logging=$logging train.iftest=$iftest


# run=2
# dataset=FACED
# gpus='[0]'
# proj_name="$dataset""_test_loo"
# valid_method='loo'
# iftest=False

# python extract_fea_forInterp.py log.run=$run log.proj_name=$proj_name data=$dataset \
#                       model.timeFilterLen=30 model.dilation_array=[1,3,6,12] \
#                       model.seg_att=15 model.avgPoolLen=15 model.timeSmootherLen=3 \
#                       train.gpus=$gpus train.valid_method=$valid_method \
#                       train.num_workers=8 \
#                       train.iftest=$iftest