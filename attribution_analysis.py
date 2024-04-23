import argparse
import config
import torch
import os
import numpy as np
from models import Conv_att_simple_new, ConvNet_complete_baseline_new, simpleNN3
from dataset import *
from train_utils import TrainFramework
import scipy.io as sio
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=config.DESCRIPTION)
    args = parser.parse_args()

    # 设定随机数种子
    args.randseed = config.RANDSEED
    # random.seed(args.randseed)
    np.random.seed(args.randseed)
    torch.manual_seed(args.randseed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # 设置线程数以及GPU
    args.gpu_index = config.GPU_INDEX
    torch.set_num_threads(8)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_index
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # # 设置模型参数

    args.model = config.MODEL
    args.n_class = config.N_CLASS
    
    # ext参数
    args.n_timeFilters = config.N_TIMEFILTERS
    args.timeFilterLen = config.TIMEFILTERLEN
    args.n_msFilters = config.N_MSFILTERS
    args.msFilterLen = config.MSFILTERLEN
    args.n_channs = config.N_CHANNS
    args.dilation_array = config.DILATION_ARRAY
    args.seg_att = config.SEG_ATT
    args.avgPoolLen = config.AVGPOOLLEN
    args.timeSmootherLen = config.TIMESMOOTHERLEN
    args.multiFact = config.MULTIFACT
    args.activ = config.ACTIV
    args.stratified = config.STRATIFIED
    args.ext_temp = config.EXT_TEMPERATURE
    args.saveFea = config.SAVEFEA
    args.has_att = config.HAS_ATT
    args.global_att = config.GLOBAL_ATT
    # mlp参数
    args.fea_dim = 4*config.N_TIMEFILTERS*config.N_MSFILTERS  # n_dialation_mode*n_tf*n_msf
    args.hidden_dim = config.HIDDEN_DIM
    args.out_dim = config.N_CLASS

    # 设置模型优化参数
    args.ext_lr = config.EXT_LR
    args.mlp_lr = config.MLP_LR
    args.ext_wd = config.EXT_WEIGHT_DECAY
    args.mlp_wd = config.MLP_WEIGHT_DECAY

    # 设置数据集相关参数
    args.fs = config.FS
    args.n_vids = config.N_VIDS
    args.n_subs = config.N_SUBS

    args.t_period =  config.T_PERIOD
    args.timeLen2 = config.TIMELEN2
    args.timestep2 = config.TIMESTEP2
    args.data_len = args.fs * args.timeLen 
    
    # mlp训练,数据特征提取设置
    args.normTrain = config.NORMTRAIN

    args.extract_mode
    args.norm_decay_rate

    # 设置数据集读取路径并加载数据，包括extract对比学习部分（5，2），和mlp预测部分（1，1）
    args.data_dir = config.DATA_DIR
    print('data_dir:', args.data_dir)

    # 设置模型保存路径
    save_dir = os.path.join(config.ROOT_DIR, config.MODEL_DIR, config.RUN_NO, 'fold0')
    if not os.path.exists(save_dir):
        print('models do not exist!')
        exit()
    print('save_dir:', save_dir)





    # 加载extractor   打印参数
    if args.model == 'att':
        model_ext = Conv_att_simple_new(args.n_timeFilters, args.timeFilterLen, args.n_msFilters, args.msFilterLen, args.n_channs, args.dilation_array, args.seg_att, 
        args.avgPoolLen, args.timeSmootherLen, args.multiFact, args.stratified,  args.activ, args.ext_temp, args.saveFea, args.has_att, args.extract_mode, args.global_att).to(device)
        print('use att model!')
    elif args.model == 'baseline':
        model_ext = ConvNet_complete_baseline_new(args.n_timeFilters, args.timeFilterLen, args.n_msFilters*4, args.msFilterLen,
                args.avgPoolLen, args.timeSmootherLen, args.n_channs, args.stratified, args.multiFact, args.saveFea, args.extract_mode).to(device)
        print('use baseline model!')
    print('model extractor:')
    print(model_ext)
    para_num = sum([p.data.nelement() for p in model_ext.parameters()])
    print('Total number of parameters:', para_num)
    
    # 加载mlp   打印参数
    model_mlp = simpleNN3(args.fea_dim, args.hidden_dim, args.out_dim).to(device)
    print('model mlp:')
    print(model_mlp)
    para_num = sum([p.data.nelement() for p in model_mlp.parameters()])
    print('Total number of parameters:', para_num)

    # 声明训练框架
    trainFramework = TrainFramework(args, model_ext, model_mlp).to(device)

    # load extractor params 
    checkpoint = torch.load(os.path.join(save_dir, 'latest_ext_cp.pth.tar'), map_location=device)
    state_dict = checkpoint['state_dict']
    trainFramework.extractor.load_state_dict(state_dict, strict=True)


    # load mlp params 
    checkpoint = torch.load(os.path.join(save_dir, 'latest_mlp_cp.pth.tar'), map_location=device)
    state_dict = checkpoint['state_dict']
    trainFramework.mlp.load_state_dict(state_dict, strict=True)

    # 不清除data，每次循环都使用这部分data
    # load mlp data
    data2, label2_repeat, n_samples2 = load_processed_FACED_data(args.data_dir, args.fs, args.n_channs, args.t_period,args.timeLen2,args.timeStep2,args.n_class)
    data2 = data2.reshape(data2.shape[0], -1, n_samples2, data2.shape[-2], data2.shape[-1])   # (train_subs,28,30, 32, 250)
    label2_repeat = label2_repeat.reshape(-1,n_samples2)
    args.n_samples2 = n_samples2

    if args.n_class == 2:
        labels = [0] * 12
        labels.extend([1] * 12)
    elif args.n_class == 9:
        labels = [0] * 3
        for i in range(1,4):
            labels.extend([i] * 3)
        labels.extend([4] * 4)
        for i in range(5,9):
            labels.extend([i] * 3)
    
    vid_inds = np.arange(args.n_vids)
    val_sub = np.arange(args.n_subs)
    args.val_sub = val_sub
    # Loop for each emotion
    attrs_all = {}
    for target_label in range(args.n_class):
        val_vid_inds = vid_inds[labels==target_label]
        args.val_vid_inds = val_vid_inds
        args.n_vids_val = len(val_vid_inds)

        

        
        data2_val_all = data2[:,args.val_vid_inds]
        label2_val_repeat = label2_repeat[args.val_vid_inds].reshape(-1)
        data2_val = data2_val_all[list(val_sub)]
        # del data2
        # del label2_repeat
        del data2_val_all
        # 特征提取前norm Train
        if args.normTrain:
            print('normTrain')
            temp = np.transpose(data2_val,(0,1,2,4,3))
            temp = temp.reshape(-1,temp.shape[-1])
            data2_mean = np.mean(temp, axis=0)
            data2_var = np.var(temp, axis=0)
            data2_val = (data2_val - data2_mean.reshape(-1,1)) / np.sqrt(data2_var + 1e-5).reshape(-1,1)
            del temp
        else:
            print('Do no norm')

        data2_val = data2_val.reshape(-1, data2_val.shape[-2], data2_val.shape[-1])
        label2_val = np.tile(label2_val_repeat, len(val_sub))
        del label2_val_repeat

        val_bs2 = args.n_vids_val *args.n_samples2
        args.mlp_val_bs =  args.n_vids_val *args.n_samples2
        valset2 = FACED_Dataset(data2_val, label2_val)
        val_loader2 = DataLoader(dataset=valset2, batch_size=val_bs2, pin_memory=True, num_workers=8,shuffle=False)
        attrs = trainFramework.attribution_analysis(val_loader2, target_label, args.mlp_val_bs)

        del data2_val
        del valset2
        del val_loader2

        if args.n_class == 2:
            if target_label == 0:
                attrs_all['negative'] = attrs
            else:
                attrs_all['positive'] = attrs
        elif args.n_class == 9:
            attrs_all['emo'+str(target_label)] = attrs
    
    sio.savemat(os.path.join(save_dir, 'attrs_all_%s.mat' % args.n_class), attrs_all)