 
import numpy as np
from data.io_utils import load_finetune_EEG_data, get_load_data_func, load_processed_SEEDV_NEW_data
from data.data_process import running_norm_onesubsession, LDS, LDS_acc
from reorder_vids import video_order_load, reorder_vids_sepVideo, reorder_vids_back
import hydra
from omegaconf import DictConfig
from model import ExtractorModel
from data.dataset import SEEDV_Dataset 
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
import os
from tqdm import tqdm
import logging

log = logging.getLogger(__name__)

@hydra.main(config_path="cfgs", config_name="config", version_base="1.3")
def ext_fea(cfg: DictConfig) -> None:
    load_dir = os.path.join(cfg.data.data_dir,'processed_data')
    data2, onesub_label2, n_samples2_onesub, n_samples2_sessions = load_finetune_EEG_data(load_dir, cfg.data)
    data2 = data2.reshape(cfg.data.n_subs, -1, data2.shape[-2], data2.shape[-1])
    save_dir = os.path.join(cfg.data.data_dir,'ext_fea',f'fea_r{cfg.log.run}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir) 
    np.save(save_dir+'/onesub_label2.npy',onesub_label2)
    
    
    
    if isinstance(cfg.train.valid_method, int):
        n_folds = cfg.train.valid_method
    elif cfg.train.valid_method == 'loo':
        n_folds = cfg.train.n_subs

    n_per = round(cfg.data.n_subs / n_folds)
    
    for fold in range(n_folds):
        log.info(f"fold:{fold}")
        if n_folds == 1:
            val_subs = []
        elif fold < n_folds - 1:
            val_subs = np.arange(n_per * fold, n_per * (fold + 1))
        else:
            val_subs = np.arange(n_per * fold, cfg.data.n_subs)            
        train_subs = list(set(np.arange(cfg.data.n_subs)) - set(val_subs))
        # if len(val_subs) == 1:
        #     val_subs = list(val_subs) + train_subs
        log.info(f'train_subs:{train_subs}')
        log.info(f'val_subs:{val_subs}' )
        


        data2_train = data2[train_subs] # (subs,vid*n_samples, 62, 1250)
        
        # print(data2[0,0])
        if cfg.ext_fea.normTrain:
            data2_fold = normTrain(data2,data2_train)
        else:
            log.info('no normTrain')
            data2_fold = data2
        # print(data2_fold[0,0])
        
        data2_fold = data2_fold.reshape(-1, data2_fold.shape[-2], data2_fold.shape[-1])
        label2_fold = np.tile(onesub_label2, cfg.data.n_subs)
        foldset = SEEDV_Dataset(data2_fold, label2_fold)
        del data2_fold, label2_fold
        fold_loader = DataLoader(foldset, batch_size=cfg.ext_fea.batch_size, shuffle=False, num_workers=cfg.train.num_workers)
        checkpoint =  os.path.join(cfg.ext_fea.cp_dir,f'f_{fold}_best.ckpt.ckpt')
        Extractor = ExtractorModel.load_from_checkpoint(checkpoint_path=checkpoint)
        Extractor.model.stratified = []
        log.info('load model:'+checkpoint)
        trainer = pl.Trainer(accelerator='gpu', devices=cfg.train.gpus)
        pred = trainer.predict(Extractor, fold_loader)
        # data
        # pred = torch.stack(pred,dim=0)
        pred = torch.cat(pred, dim=0).cpu().numpy()
        log.debug(pred.shape)
        # pred = pred
        
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
            log.warning("There are inf values in the array")
        else:
            log.info('no inf')
        if np.isnan(fea).any():
            log.warning("There are nan values in the array")
        else:
            log.info('no nan')
            
        # reorder
        if cfg.data.dataset_name == 'FACED':
            vid_order = video_order_load(cfg.data.n_vids)
            if cfg.data.n_class == 2:
                n_vids = 24
            elif cfg.data.n_class == 9:
                n_vids = 28
            vid_inds = np.arange(n_vids)
            fea, vid_play_order_new = reorder_vids_sepVideo(fea, vid_order, vid_inds, n_vids)


        n_sample_sum_sessions = np.sum(n_samples2_sessions,1)
        n_sample_sum_sessions_cum = np.concatenate((np.array([0]), np.cumsum(n_sample_sum_sessions)))

        # fea_processed = np.zeros_like(fea)
        log.info('running norm:')
        for sub in range(cfg.data.n_subs):
            log.info(f'sub:{sub}')
            for s in  tqdm(range(len(n_sample_sum_sessions)), desc='sessions', leave=False):
                fea[sub,n_sample_sum_sessions_cum[s]:n_sample_sum_sessions_cum[s+1]] = running_norm_onesubsession(
                                            fea[sub,n_sample_sum_sessions_cum[s]:n_sample_sum_sessions_cum[s+1]],
                                            data_mean,data_var,cfg.ext_fea.rn_decay)
                
                tqdm.set_postfix({'sub': s})
        # print('rn:',fea[0,0])
        if np.isinf(fea).any():
            log.warning("There are inf values in the array")
        else:
            log.info('no inf')
        if np.isnan(fea).any():
            log.warning("There are nan values in the array")
        else:
            log.info('no nan')

        # order back
        if cfg.data.dataset_name == 'FACED':
            fea = reorder_vids_back(fea, len(vid_inds), vid_play_order_new)
        
        n_samples2_onesub_cum = np.concatenate((np.array([0]), np.cumsum(n_samples2_onesub)))
        # LDS
        log.info('LDS:')
        for sub in range(cfg.data.n_subs):
            log.info(f'sub:{sub}')
            for vid in tqdm(range(len(n_samples2_onesub)), desc='vids', leave=False):
                fea[sub,n_samples2_onesub_cum[vid]:n_samples2_onesub_cum[vid+1]] = LDS(fea[sub,n_samples2_onesub_cum[vid]:n_samples2_onesub_cum[vid+1]])
            # print('LDS:',fea[sub,0])
                tqdm.set_postfix({'vid': vid})
        fea = fea.reshape(-1,fea.shape[-1])
        
        
        # (8.32145433e-18-8.31764020e-18)/np.sqrt(4.01888196e-40)
        
        # max_fea = np.max(fea)
        # min_fea = np.min(fea)
        # print(max_fea,min_fea)
        if np.isinf(fea).any():
            log.warning("There are inf values in the array")
        else:
            log.info('no inf')
        if np.isnan(fea).any():
            log.warning("There are nan values in the array")
        else:
            log.info('no nan')

        save_path = os.path.join(save_dir,f'fold_{fold}_fea_'+cfg.ext_fea.mode+'.npy')
        # if not os.path.exists(cfg.ext_fea.save_dir):
        #     os.makedirs(cfg.ext_fea.save_dir)  
        np.save(save_path,fea)
        
        if cfg.train.iftest :
            break

    
def normTrain(data2,data2_train):
    log.info('normTrain')
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