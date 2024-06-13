run=44
seg_att=15
msFilterLen=3
ext_wd=0.00015
# mlp_wds=(0.001 0.0022 0.005 0.011 0.025 0.056 0.125)
mlp_wds=(0.001 0.0022 0.005 0.0075 0.011)
gpus='[3]'


dataset=FACED_def_c2
valid_method=10
logging=default
iftest=False
proj_name="$dataset""_epoch30"
# exp_name="segatt$seg_att"
# exp_name="mslen$msFilterLen"
# exp_name="ext_epoch30"
# exp_name="extwd$ext_wd"
exp_name="s$seg_att""_mslen$msFilterLen"


# you can run it in once
echo "proj: $proj_name exp:$exp_name run: $run gpus: $gpus valid_method: $valid_method"
echo "train ext wd: $ext_wd"
python train_ext.py log.run=$run log.proj_name=$proj_name data=$dataset \
                    train.gpus=$gpus train.valid_method=$valid_method \
                    hydra/job_logging=$logging train.iftest=$iftest \
                    log.exp_name=$exp_name  \
                    model.seg_att=$seg_att\
                    model.msFilterLen=$msFilterLen\
                    train.wd=$ext_wd \
                    # model.timeFilterLen=60 model.msFilterLen=3 model.dilation_array=[1,6,12,24] \
                    # model.seg_att=30 model.avgPoolLen=30 model.timeSmootherLen=6
echo "extract fea with wd: $ext_wd"
python extract_fea.py log.run=$run log.proj_name=$proj_name data=$dataset \
                      train.gpus=$gpus train.valid_method=$valid_method \
                      hydra/job_logging=$logging train.iftest=$iftest \
                      log.exp_name=$exp_name \
                      model.seg_att=$seg_att\
                      model.msFilterLen=$msFilterLen\
                      train.wd=$ext_wd \
                    #   model.timeFilterLen=60 model.msFilterLen=3 model.dilation_array=[1,6,12,24] \
                    #   model.seg_att=30 model.avgPoolLen=30 model.timeSmootherLen=6

for mlp_wd in "${mlp_wds[@]}"
do
  # exp_name="extwd$ext_wd""_mlpwd$mlp_wd"
  echo "exp_name: $exp_name"
  echo "Training MLP with mlp_wd: $mlp_wd and ext_wd: $ext_wd"
  python train_mlp.py log.run=$run log.proj_name=$proj_name data=$dataset \
                    train.gpus=$gpus train.valid_method=$valid_method \
                    hydra/job_logging=$logging train.iftest=$iftest \
                    log.exp_name=$exp_name \
                    model.seg_att=$seg_att\
                    model.msFilterLen=$msFilterLen\
                    train.wd=$ext_wd \
                    mlp.wd=$mlp_wd \
                    # model.timeFilterLen=60 model.msFilterLen=3 model.dilation_array=[1,6,12,24] \
                    # model.seg_att=30 model.avgPoolLen=30 model.timeSmootherLen=6
done