 
import numpy as np
from data.io_utils import load_processed_SEEDV_data, load_processed_SEEDV_NEW_data
from data.data_process import running_norm_onesub, LDS, LDS_acc
import hydra
from omegaconf import DictConfig
from model import ExtractorModel
from data.dataset import SEEDV_Dataset 
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
import os
from tqdm import tqdm

@hydra.main(config_path="cfgs", config_name="config", version_base="1.3")
def ext_fea(cfg: DictConfig) -> None:
    # print(cfg.mlp.fea_dim, cfg.mlp.hidden_dim, cfg.mlp.out_dim)
    data2, onesub_label2, n_samples2_onesub, n_samples2_sessions = load_processed_SEEDV_NEW_data(
            cfg.data.load_dir, cfg.data.fs, cfg.data.n_channs, cfg.data.timeLen2,cfg.data.timeStep2,cfg.data.n_session,cfg.data.n_subs,cfg.data.n_vids,cfg.data.n_class)
    data2 = data2.reshape(cfg.data.n_subs, -1, data2.shape[-2], data2.shape[-1])
    if not os.path.exists(cfg.ext_fea.save_dir):
        os.makedirs(cfg.ext_fea.save_dir) 
    np.save(cfg.ext_fea.save_dir+'/onesub_label2.npy',onesub_label2)
    
    
    
    if cfg.train.valid_method == '5-folds':
        n_folds = 5
    elif cfg.train.valid_method == 'loo':
        n_folds = cfg.train.n_subs
    elif cfg.train.valid_method == 'all':
        n_folds = 1

    n_per = round(cfg.data.n_subs / n_folds)
    
    for fold in range(n_folds):
        print("fold:", fold)
        if n_folds == 1:
            val_subs = []
        elif fold < n_folds - 1:
            val_subs = np.arange(n_per * fold, n_per * (fold + 1))
        else:
            val_subs = np.arange(n_per * fold, cfg.data.n_subs)            
        train_subs = list(set(np.arange(cfg.data.n_subs)) - set(val_subs))
        # if len(val_subs) == 1:
        #     val_subs = list(val_subs) + train_subs
        print('train_subs:', train_subs)
        print('val_subs:', val_subs)
        


        data2_train = data2[train_subs] # (subs,vid*n_samples, 62, 1250)
        
        # print(data2[0,0])
        if cfg.ext_fea.normTrain:
            data2_fold = normTrain(data2,data2_train)
        else:
            print('no normTrain')
            data2_fold = data2
        # print(data2_fold[0,0])
        
        data2_fold = data2_fold.reshape(-1, data2_fold.shape[-2], data2_fold.shape[-1])
        label2_fold = np.tile(onesub_label2, cfg.data.n_subs)
        foldset = SEEDV_Dataset(data2_fold, label2_fold)
        del data2_fold, label2_fold
        fold_load = DataLoader(foldset, batch_size=cfg.ext_fea.batch_size, shuffle=False, num_workers=cfg.train.num_workers)
        checkpoint =  cfg.ext_fea.cp_dir+f'f_{fold}_best.ckpt.ckpt'
        Extractor = ExtractorModel.load_from_checkpoint(checkpoint_path=checkpoint)
        Extractor.model.stratified = []
        print('load model:', checkpoint)
        trainer = pl.Trainer(accelerator='gpu', devices=cfg.train.gpus)
        pred = trainer.predict(Extractor, fold_load)
        # data
        pred = torch.stack(pred,dim=0)
        pred = pred.reshape(-1,pred.shape[-3],pred.shape[-2],pred.shape[-1]).cpu().numpy()
        
        # max_fea = np.max(pred)
        # min_fea = np.min(pred)
        # print(max_fea,min_fea)
        # if np.isinf(pred).any():
        #     print("There are inf values in the array")
        
        fea = cal_fea(pred,cfg.ext_fea.mode)
        # print('fea0:',fea[0])


        
        fea = fea.reshape(cfg.data.n_subs,-1,fea.shape[-1])
        fea_train = fea[train_subs]
        
        data_mean = np.mean(np.mean(fea_train, axis=1),axis=0)
        data_var = np.mean(np.var(fea_train, axis=1),axis=0)
        # print('fea_mean:',data_mean) 
        # print('fea_var:',data_var)
        if np.isinf(fea).any():
            print("There are inf values in the array")
        else:
            print('no inf')
        if np.isnan(fea).any():
            print("There are nan values in the array")
        else:
            print('no nan')


        n_sample_sum_sessions = np.sum(n_samples2_sessions,1)
        n_sample_sum_sessions_cum = np.concatenate((np.array([0]), np.cumsum(n_sample_sum_sessions)))

        # fea_processed = np.zeros_like(fea)
        print('running norm:')
        for sub in range(cfg.data.n_subs):
            print('sub:',sub)
            for s in  tqdm(range(len(n_sample_sum_sessions)), desc='sessions', leave=False):
                fea[sub,n_sample_sum_sessions_cum[s]:n_sample_sum_sessions_cum[s+1]] = running_norm_onesub(
                        fea[sub,n_sample_sum_sessions_cum[s]:n_sample_sum_sessions_cum[s+1]],data_mean,data_var,cfg.ext_fea.rn_decay)
        # print('rn:',fea[0,0])
        if np.isinf(fea).any():
            print("There are inf values in the array")
        else:
            print('no inf')
        if np.isnan(fea).any():
            print("There are nan values in the array")
        else:
            print('no nan')
        
        n_samples2_onesub_cum = np.concatenate((np.array([0]), np.cumsum(n_samples2_onesub)))
        # LDS
        print('LDS:')
        for sub in range(cfg.data.n_subs):
            print('sub:',sub)
            for vid in tqdm(range(len(n_samples2_onesub)), desc='vids', leave=False):
                fea[sub,n_samples2_onesub_cum[vid]:n_samples2_onesub_cum[vid+1]] = LDS(fea[sub,n_samples2_onesub_cum[vid]:n_samples2_onesub_cum[vid+1]])
            # print('LDS:',fea[sub,0])
        fea = fea.reshape(-1,fea.shape[-1])
        
        
        # (8.32145433e-18-8.31764020e-18)/np.sqrt(4.01888196e-40)
        
        # max_fea = np.max(fea)
        # min_fea = np.min(fea)
        # print(max_fea,min_fea)
        if np.isinf(fea).any():
            print("There are inf values in the array")
        else:
            print('no inf')
        if np.isnan(fea).any():
            print("There are nan values in the array")
        else:
            print('no nan')

        save_path = os.path.join(cfg.ext_fea.save_dir,f'fold_{fold}_fea_'+cfg.ext_fea.mode+'.npy')
        # if not os.path.exists(cfg.ext_fea.save_dir):
        #     os.makedirs(cfg.ext_fea.save_dir)  
        np.save(save_path,fea)

    
def normTrain(data2,data2_train):
    print('normTrain')
    temp = np.transpose(data2_train,(0,1,3,2))
    temp = temp.reshape(-1,temp.shape[-1])
    data2_mean = np.mean(temp, axis=0)
    data2_var = np.var(temp, axis=0)
    data2_normed = (data2 - data2_mean.reshape(-1,1)) / np.sqrt(data2_var + 1e-5).reshape(-1,1)
    return data2_normed

def cal_fea(data,mode):
    if mode == 'de':
        # print(np.var(data, 3).squeeze()[0])
        fea = 0.5*np.log(2*np.pi*np.exp(1)*(np.var(data, 3))).squeeze()
        fea[fea<-40] = -40
    elif mode == 'me':
        fea = np.mean(data, axis=3).squeeze()
    # print(fea.shape)
    # print(fea[0])
    return fea

if __name__ == '__main__':
    ext_fea()