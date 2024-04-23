from model.models import Conv_att_timefilter
import random
import numpy as np
import torch
import config
import argparse
import os
from data.dataset import *
import hdf5storage


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=config.DESCRIPTION)
    
    args = parser.parse_args()

    # # 设置模型参数

    args.model = config.MODEL
    args.n_class = config.N_CLASS

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

    # 设置数据集相关参数
    args.fs = config.FS
    args.n_vids = config.N_VIDS
    args.t_period =  config.T_PERIOD
    args.n_subs = config.N_SUBS



    args.n_subs = config.N_SUBS
    args.extract_mode = config.EXTRACT_MODE

    args.data_dir = config.DATA_DIR
    print('data_dir:', args.data_dir)

    args.randseed = config.RANDSEED
    random.seed(args.randseed)
    np.random.seed(args.randseed)
    torch.manual_seed(args.randseed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args.bs = config.BATCH_SIZE

    # 设置线程数以及GPU
    args.gpu_index = config.GPU_INDEX
    torch.set_num_threads(8)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_index
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = os.path.join(config.ROOT_DIR, config.MODEL_DIR, config.RUN_NO, 'fold0')
    if not os.path.exists(save_dir):
        print('models do not exist!')
        exit()
    print('save_dir:', save_dir)



    model_timefilter = Conv_att_timefilter(args.n_timeFilters, args.timeFilterLen, args.n_msFilters, args.msFilterLen, args.n_channs, args.dilation_array, args.seg_att, 
    args.avgPoolLen, args.timeSmootherLen, args.multiFact, args.stratified,  args.activ, args.ext_temp, args.saveFea, args.has_att, args.extract_mode, args.global_att).to(device)
    checkpoint = torch.load(os.path.join(save_dir, 'latest_ext_cp.pth.tar'), map_location=device)
    state_dict = checkpoint['state_dict']
    model_timefilter.load_state_dict(state_dict, strict=True)
    print('load latest_ext_model!')


    data, label_repeat, n_samples = load_processed_FACED_data(args.data_dir, args.fs, args.n_channs, args.t_period,args.t_period,args.t_period,args.n_class)
    print(data.shape)
    data = data.reshape(-1,data.shape[-2],data.shape[-1])
    print(data.shape)
    label_train = np.tile(label_repeat, args.n_subs)
    trainset = FACED_Dataset(data, label_train)
    train_loader = DataLoader(dataset=trainset, batch_size=args.bs, pin_memory=True, num_workers=8,shuffle=False)
    model_timefilter.eval()
    if args.timeFilterLen%2 == 0:
        # data_all = np.empty((data.shape[0],args.n_timeFilters,data.shape[-2],data.shape[-1]-1),float)
        data_all = torch.rand(data.shape[0],args.n_timeFilters,data.shape[-2],data.shape[-1]-1)
    pos = 0
    for cnt,(x,y) in enumerate(train_loader):
        bs = x.shape[0]
        x = x.to(device)
        y = y.to(device)
        timefiltered_fea = model_timefilter(x)
        data_all[pos:pos+bs] = timefiltered_fea
        pos += bs
    print(pos)
    print(data_all.shape)
    data_all = data_all.cpu().detach().numpy()


    data_all = np.transpose(data_all,(1,0,3,2))
    data_all = data_all.reshape(data_all.shape[0],args.n_subs,-1,data_all.shape[-1])


    print(data_all.shape)
    data_all = np.transpose(data_all,(0,1,3,2))
    print(data_all.shape)

    data = {'data_all': data_all,}
    


    hdf5storage.savemat(os.path.join(save_dir, 'timefiltered_data_all_mslen%d.mat' % args.msFilterLen), data, do_compression=True, format='7.3')

