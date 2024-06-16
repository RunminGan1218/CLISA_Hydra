import hydra
from omegaconf import DictConfig
from model.models import simpleNN3
import numpy as np
import os
from data.dataset import PDataset
from model.pl_models import MLPModel
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
import torch
import logging

log = logging.getLogger(__name__)

@hydra.main(config_path="cfgs", config_name="config", version_base="1.3")
def train_mlp(cfg: DictConfig) -> None:
    
    pl.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if isinstance(cfg.train.valid_method, int):
        n_folds = cfg.train.valid_method
    elif cfg.train.valid_method == 'loo':
        n_folds = cfg.train.n_subs

    n_per = round(cfg.data.n_subs / n_folds)
    best_val_acc_list = []
    
    for fold in range(0,n_folds):
        cp_dir = os.path.join(cfg.log.cp_dir, cfg.data.dataset_name)
        wandb_logger = WandbLogger(name=cfg.log.exp_name+'mlp'+'v'+str(cfg.train.valid_method)
                                   +f'_{cfg.data.timeLen}_{cfg.data.timeStep}_r{cfg.log.run}'+f'_f{fold}', 
                                   project=cfg.log.proj_name, log_model="all")
        checkpoint_callback = ModelCheckpoint(monitor="mlp/val/acc", mode="max", dirpath=cp_dir, filename=cfg.log.exp_name+'_mlp_r'+str(cfg.log.run)+f'_f{fold}_best_wd{cfg.mlp.wd}')
        earlyStopping_callback = EarlyStopping(monitor="mlp/val/acc", mode="max", patience=cfg.mlp.patience)
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
        log.info(f'val_subs:{val_subs}')
        
        save_dir = os.path.join(cfg.data.data_dir,'ext_fea',f'fea_r{cfg.log.run}')
        save_path = os.path.join(save_dir,cfg.log.exp_name+'_r'+str(cfg.log.run)+f'_f{fold}_fea_'+cfg.ext_fea.mode+'.npy')
        data2 = np.load(save_path)
        # print(data2[:,160])
        if np.isnan(data2).any():
            log.warning('nan in data2')
            data2 = np.where(np.isnan(data2), 0, data2)
        data2 = data2.reshape(cfg.data.n_subs, -1, data2.shape[-1])
        onesub_label2 = np.load(save_dir+'/onesub_label2.npy')
        labels2_train = np.tile(onesub_label2, len(train_subs))
        labels2_val = np.tile(onesub_label2, len(val_subs))
        trainset2 = PDataset(data2[train_subs].reshape(-1,data2.shape[-1]), labels2_train)
        # trainset2 = PDataset(data2[val_subs].reshape(-1,data2.shape[-1]), labels2_val)
        valset2 = PDataset(data2[val_subs].reshape(-1,data2.shape[-1]), labels2_val)
        trainLoader = DataLoader(trainset2, batch_size=cfg.mlp.batch_size, shuffle=True, num_workers=cfg.mlp.num_workers)
        valLoader = DataLoader(valset2, batch_size=cfg.mlp.batch_size, shuffle=False, num_workers=cfg.mlp.num_workers)
        model_mlp = simpleNN3(cfg.mlp.fea_dim, cfg.mlp.hidden_dim, cfg.mlp.out_dim,0.1)
        predictor = MLPModel(model_mlp, cfg.mlp)
        trainer = pl.Trainer(logger=wandb_logger, callbacks=[checkpoint_callback, earlyStopping_callback],max_epochs=cfg.mlp.max_epochs, min_epochs=cfg.mlp.min_epochs, accelerator='gpu', devices=cfg.mlp.gpus)
        trainer.fit(predictor, trainLoader, valLoader)
        best_val_acc_list.append(checkpoint_callback.best_model_score.item())
        wandb.finish()
        
        if cfg.train.iftest :
            break

    log.info("Best validation accuracies for each fold:")
    for fold, acc in enumerate(best_val_acc_list):
        log.info(f"    Fold {fold}: {acc}")
    
    average_val_acc = np.mean(best_val_acc_list)
    log.info(f"Average validation accuracy across all folds: {average_val_acc}")
    std_val_acc = np.std(best_val_acc_list)
    log.info(f"Standard deviation of validation accuracy across all folds: {std_val_acc}")
    log.info(f"Extracting features with {cfg.mlp.wd}: $mlp_wd and ext_wd: {cfg.train.wd}")

if __name__ == '__main__':
    train_mlp()