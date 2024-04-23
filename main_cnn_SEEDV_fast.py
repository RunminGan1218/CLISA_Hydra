import config
import argparse
import numpy as np
import torch
import os
from torch.utils.data import DataLoader
from data.dataset import *
from model.models import Conv_att_simple_new, ConvNet_complete_baseline_new, simpleNN3
from train_utils import TrainFramework
import random
import json
from io_utils import *



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=config.DESCRIPTION)
    
    args = parser.parse_args()

    # # 设置模型参数

    args.model = config.MODEL
    
    
    # ext参数
    args.n_timeFilters = config.N_TIMEFILTERS
    args.timeFilterLen = config.TIMEFILTERLEN
    args.n_msFilters = config.N_MSFILTERS
    args.msFilterLen = config.MSFILTERLEN           
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


    

    # 训练相关参数
    args.max_tol = config.MAX_TOL
    args.rounds = config.ROUNDS
    args.epoch_ext = config.EPOCH_EXT
    args.epoch_mlp = config.EPOCH_MLP
    args.extract_mode = config.EXTRACT_MODE
    args.norm_decay_rate = config.NORM_DECAY_RATE
    args.loss_temp = config.LOSS_TEMPERATURE
    args.Lambda1 = config.LAMBDA1    # self-superviced CL loss cof
    args.Lambda2 = config.LAMBDA2    # superviced CL loss cof
    args.para_evolve1 = config.PARA_EVOLVE1
    args.para_evolve2 = config.PARA_EVOLVE2
    args.valid_method = config.VALID_METHOD


    # 设置数据集相关参数
    args.n_class = config.N_CLASS
    args.fs = config.FS
    args.n_channs = config.N_CHANNS 
    args.n_vids = config.N_VIDS
    args.t_period =  config.T_PERIOD       
    args.timeLen = config.TIMELEN           
    args.timeStep = config.TIMESTEP          
    args.data_len = args.fs * args.timeLen
    args.n_subs = config.N_SUBS
    args.timeLen2 = config.TIMELEN2
    args.timeStep2 = config.TIMESTEP2
    args.n_session = config.N_SESSION  #   config设置
    
    # 泛化性测试设置
    args.generalizationTest = config.GENERALIZATIONTEST
    
    # mlp训练,数据特征提取设置
    args.normTrain = config.NORMTRAIN

    # normal training mode 设置
    args.training_mode = config.TRAINING_MODE
    args.use_best_pretrain = config.USE_BEST_PRETRAIN
    args.max_tol_normal_ext = config.MAX_TOL_NORMAL_EXT
    args.max_tol_normal_mlp = config.MAX_TOL_NORMAL_MLP
    args.epoch_ext_normal = config.EPOCH_EXT_NORMAL
    args.epoch_mlp_normal = config.EPOCH_MLP_NORMAL



    # save设置
    args.save_latest = config.SAVE_LATEST

    # 设定随机数种子
    args.randseed = config.RANDSEED
    random.seed(args.randseed)
    np.random.seed(args.randseed)
    torch.manual_seed(args.randseed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 设置线程数以及GPU
    args.gpu_index = config.GPU_INDEX
    torch.set_num_threads(8)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_index
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 设置交叉验证训练方法
    if args.valid_method == '10-folds':
        n_folds = 10
    elif args.valid_method == 'loo':
        n_folds = args.n_subs
    elif args.valid_method == 'all':
        n_folds = 1

    n_per = round(args.n_subs / n_folds)

    # 确认stratified
    for pos in args.stratified:
        assert pos in ['initial', 'middle1', 'middle2', 'final', 'final_batch', 'middle1_batch', 'middle2_batch', 'no']


    # 设置数据集读取路径并加载数据，包括extract对比学习部分（5，2），和mlp预测部分（1，1）
    args.data_dir = config.DATA_DIR          
    print('data_dir:', args.data_dir)


    # 泛化性测试 数据集调整
    if args.generalizationTest:
        val_vid_inds = config.VAL_VID_INDS               #     ---------------改
        train_vid_inds = np.array(list(set(np.arange(args.n_vids)) - set(val_vid_inds)))
    else:
        val_vid_inds = np.arange(args.n_vids)
        train_vid_inds = np.arange(args.n_vids)
    train_vid_inds = list(train_vid_inds)
    val_vid_inds = list(val_vid_inds)
    args.train_vid_inds = train_vid_inds
    args.val_vid_inds = val_vid_inds
    args.n_vids_train = len(train_vid_inds)
    args.n_vids_val = len(val_vid_inds)





    # 设置模型保存路径
    save_dir = os.path.join(config.ROOT_DIR, config.MODEL_DIR, config.RUN_NO)       
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print('save_dir:', save_dir)

    print(parser.description)
    print(args)
    print('writing args to file...')
    argsDict = args.__dict__
    with open(os.path.join(save_dir,'config.txt'), 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')

    train_acc_list = np.zeros(n_folds)
    val_acc_list = np.zeros(n_folds)



    for fold in range(0,n_folds):
        print('fold:',fold)

        # 根据fold调整训练集和验证集
        if fold < n_folds - 1:
            val_sub = np.arange(n_per * fold, n_per * (fold + 1))
        else:
            val_sub = np.arange(n_per * fold, args.n_subs)
        
        if n_folds == 1:
            val_sub = []
        train_sub = list(set(np.arange(args.n_subs)) - set(val_sub))
        args.train_sub = train_sub
        args.val_sub = val_sub
        print('train sub:',args.train_sub)
        print('val sub:',args.val_sub)


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

        # 设置optimizer和scheduler
        opt_ext = torch.optim.Adam(model_ext.parameters(), lr=args.ext_lr, weight_decay=args.ext_wd)
        # sch_ext = torch.optim.lr_scheduler.StepLR(opt_ext, step_size=args.epochs_pretrain, gamma=0.8, last_epoch=-1, verbose=False)

        # 加载mlp   打印参数
        model_mlp = simpleNN3(args.fea_dim, args.hidden_dim, args.out_dim).to(device)
        print('model mlp:')
        print(model_mlp)
        para_num = sum([p.data.nelement() for p in model_mlp.parameters()])
        print('Total number of parameters:', para_num)

        # 设置optimizer和scheduler
        opt_mlp = torch.optim.Adam(model_mlp.parameters(), lr=args.mlp_lr, weight_decay=args.mlp_wd)
        # sch_mlp = torch.optim.lr_scheduler.StepLR(opt_mlp, step_size=args.epochs_finetune, gamma=0.8, last_epoch=-1, verbose=False)


        # 声明训练框架
        log_dir = os.path.join(save_dir,f'fold{fold}')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        trainFramework = TrainFramework(args, model_ext, model_mlp, opt_ext, opt_mlp,log_dir,temperature=args.loss_temp).to(device)



        # # 设置训练集和测试集，包括extract和mlp部分 
        # extract部分 
        data, onesub_label, n_samples_onesub, n_samples_sessions = load_processed_SEEDV_data(
            args.data_dir, args.fs, args.n_channs, args.timeLen,args.timeStep,args.n_session,args.n_subs,args.n_vids,args.n_class)
        # data.shape = (15*(sum(n_samples_onesub)))*62*point_len       (n_sub*sum_samples_onesub)*62*1250
        print('load ext data finished!')
        
        data = data.reshape(args.n_subs, -1, data.shape[-2],data.shape[-1])   # (subs,n_vids*n_samples,n_channs,n_points)

        # extract对比学习部分（5，2）
        data_train = data[list(train_sub)].reshape(-1, data.shape[-2], data.shape[-1])   # (train_subs*n_vids*30, 32, 250)
        data_val = data[list(val_sub)].reshape(-1, data.shape[-2], data.shape[-1])
        label_train = np.tile(onesub_label, len(train_sub))
        label_val = np.tile(onesub_label, len(val_sub))
        trainset = FACED_Dataset(data_train, label_train)
        valset = FACED_Dataset(np.concatenate((data_val,data_train),0), np.concatenate((label_val,label_train),0))
        del data
        del onesub_label


        train_sampler = TrainSampler_SEEDV(n_subs=len(train_sub), batch_size=args.n_vids_train,
                                            n_samples_session=n_samples_sessions, n_session=args.n_session, n_times=1)
        val_sampler = TrainSampler_SEEDV(n_subs=len(val_sub)+len(train_sub), batch_size=args.n_vids_val,
                                        n_samples_session=n_samples_sessions, n_session=args.n_session, n_times=1,if_val_loo=True)
        # print(val_sampler.__len__())

        train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, pin_memory=True, num_workers=8)
        val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, pin_memory=True, num_workers=8)


        # 训练extractor
        trainFramework.extractor_train_fn_normal(train_loader, val_loader, args.epoch_ext_normal)
        if args.use_best_pretrain:
            print('use best pretrain!')
            checkpoint = torch.load(os.path.join(log_dir, 'best_ext_cp.pth.tar'), map_location=device)
            state_dict = checkpoint['state_dict']
            trainFramework.extractor.load_state_dict(state_dict, strict=True)
        else:
            print('use latest pretrain!')
        del data_train
        del data_val
        del trainset
        del valset
        del train_loader
        del val_loader

        # 训练mlp部分
        # data2, label2_repeat, n_samples2 = load_processed_FACED_data(args.data_dir, args.fs, args.n_channs, args.t_period,args.timeLen2,args.timeStep2,args.n_class)
        data2, onesub_label2, n_samples2_onesub, n_samples2_sessions = load_processed_SEEDV_data(
            args.data_dir, args.fs, args.n_channs, args.timeLen2,args.timeStep2,args.n_session,args.n_subs,args.n_vids,args.n_class)
        print('load mlp data finished!')

        data2 = data2.reshape(args.n_subs, -1, data2.shape[-2], data2.shape[-1])   # (subs,vid*n_samples, 62, 1250)

        args.n_samples2_onesub = n_samples2_onesub
        args.n_samples2_sessions = n_samples2_sessions
        
        # mlp预测部分（1，1）
        data2_train = data2[list(train_sub)]
        data2_val = data2[list(val_sub)]
        del data2


        # 特征提取前norm Train
        if args.normTrain:
            print('normTrain')
            temp = np.transpose(data2_train,(0,1,3,2))
            temp = temp.reshape(-1,temp.shape[-1])
            data2_mean = np.mean(temp, axis=0)
            data2_var = np.var(temp, axis=0)
            data2_train = (data2_train - data2_mean.reshape(-1,1)) / np.sqrt(data2_var + 1e-5).reshape(-1,1)
            data2_val = (data2_val - data2_mean.reshape(-1,1)) / np.sqrt(data2_var + 1e-5).reshape(-1,1)
            del temp
        else:
            print('Do no norm')

        data2_train = data2_train.reshape(-1, data2_train.shape[-2], data2_train.shape[-1])
        data2_val = data2_val.reshape(-1, data2_val.shape[-2], data2_val.shape[-1])
        label2_train = np.tile(onesub_label2, len(train_sub))
        label2_val = np.tile(onesub_label2, len(val_sub))
        del onesub_label2

        train_bs2 = args.n_vids_train *int(np.mean(n_samples2_onesub))
        trainset2 = FACED_Dataset(data2_train, label2_train)
        train_loader2 = DataLoader(dataset=trainset2, batch_size=train_bs2, pin_memory=True, num_workers=8,shuffle=False)
        val_bs2 = max(min(args.n_vids_val *int(np.mean(n_samples2_onesub)),len(label2_val)),1)
        valset2 = FACED_Dataset(data2_val, label2_val)
        val_loader2 = DataLoader(dataset=valset2, batch_size=val_bs2, pin_memory=True, num_workers=8,shuffle=False)

        args.trainset_len2 = trainset2.__len__()
        args.valset_len2 = valset2.__len__()

        # mlp训练相关参数 
        args.mlp_train_bs = args.n_vids_train *int(np.mean(n_samples2_onesub))
        args.mlp_val_bs =  max(min(args.n_vids_val *int(np.mean(n_samples2_onesub)),len(label2_val)),1)
        val_acc_list[fold], best_loss, train_acc_list[fold], best_train_loss = trainFramework.mlp_train_fn_normal_SEEDV(
                                                            train_loader2, val_loader2, args.epoch_mlp_normal)
        
        del data2_train
        del data2_val
        del trainset2
        del valset2
        del train_loader2
        del val_loader2

        if args.save_latest == True:
            save_checkpoint(trainFramework.extractor,trainFramework.opt_ext,save_path=os.path.join(log_dir, 'latest_ext_cp.pth.tar'))
            save_checkpoint(trainFramework.mlp,trainFramework.opt_mlp,save_path=os.path.join(log_dir, 'latest_mlp_cp.pth.tar'))



    ave_train_acc = np.mean(train_acc_list)
    ave_val_acc = np.mean(val_acc_list)
    print('average train acc:',ave_train_acc,'average val acc:',ave_val_acc)

    for k,v in config.EXPERIMENT.items():
        print(k)
        print(json.dumps(v, indent=4))
    print('finished!')


