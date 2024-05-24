run=29
seg_att=15
msFilterLen=7
gpus='[7]'

dataset=FACED_def
valid_method=10
logging=default
iftest=False
proj_name="$dataset"
# exp_name="segatt$seg_att""_5_2"
exp_name="mslen$msFilterLen"

# you can run it in once
echo "proj: $proj_name exp:$exp_name run: $run gpus: $gpus valid_method: $valid_method"
python train_ext.py log.run=$run log.proj_name=$proj_name data=$dataset \
                    train.gpus=$gpus train.valid_method=$valid_method \
                    hydra/job_logging=$logging train.iftest=$iftest \
                    log.exp_name=$exp_name  \
                    model.seg_att=$seg_att\
                    model.msFilterLen=$msFilterLen\
                    # model.timeFilterLen=60 model.msFilterLen=3 model.dilation_array=[1,6,12,24] \
                    # model.seg_att=30 model.avgPoolLen=30 model.timeSmootherLen=6

python extract_fea.py log.run=$run log.proj_name=$proj_name data=$dataset \
                      train.gpus=$gpus train.valid_method=$valid_method \
                      hydra/job_logging=$logging train.iftest=$iftest \
                      log.exp_name=$exp_name \
                      model.seg_att=$seg_att\
                      model.msFilterLen=$msFilterLen\
                    #   model.timeFilterLen=60 model.msFilterLen=3 model.dilation_array=[1,6,12,24] \
                    #   model.seg_att=30 model.avgPoolLen=30 model.timeSmootherLen=6

python train_mlp.py log.run=$run log.proj_name=$proj_name data=$dataset \
                    train.gpus=$gpus train.valid_method=$valid_method \
                    hydra/job_logging=$logging train.iftest=$iftest \
                    log.exp_name=$exp_name \
                    model.seg_att=$seg_att\
                    model.msFilterLen=$msFilterLen\
                    # model.timeFilterLen=60 model.msFilterLen=3 model.dilation_array=[1,6,12,24] \
                    # model.seg_att=30 model.avgPoolLen=30 model.timeSmootherLen=6