
import hydra
from omegaconf import DictConfig
import numpy as np 
from train_ext import ExtractorModel, SEEDVDataModule
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from data.dataset import SEEDV_Dataset, TrainSampler_SEEDV
from data.io_utils  import load_processed_SEEDV_data
from torch.utils.data import DataLoader
import torch

checkpoint = "/home/gpt/grm/CLISA_Hydra/runs/2024-04-19/13-47-08_/lightning_model_v1/checkpoints/f_1_best.ckpt.ckpt"

@hydra.main(config_path="cfgs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    
    
    # # checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
    # # print(checkpoint.keys())
    # # print(checkpoint["hyper_parameters"])
    pl.seed_everything(cfg.seed)
    Extractor = ExtractorModel.load_from_checkpoint(checkpoint_path=checkpoint)
    # print(Extractor)
    val_subs = [1]
    train_subs = list(set(np.arange(cfg.data.n_subs)) - set(val_subs))
    if len(val_subs) == 1:
        val_subs = list(val_subs) + train_subs
    print(train_subs)
    print(val_subs)
    wandb_logger = WandbLogger(name=f'test', project=cfg.log.proj_name)
    dm = SEEDVDataModule(cfg.data.load_dir,cfg.data.save_dir, cfg.data.timeLen, cfg.data.timeStep, train_subs, val_subs, cfg.train.train_vids, cfg.train.val_vids, cfg.data.n_session, cfg.train.valid_method=='loo', cfg.train.num_workers)
    trainer = pl.Trainer(logger=wandb_logger, accelerator='gpu', devices=cfg.train.gpus)
    trainer.validate(Extractor, dm)
    
@hydra.main(config_path="cfgs", config_name="config", version_base="1.3")
def main2(cfg: DictConfig) -> None:
    
    
    # # checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
    # # print(checkpoint.keys())
    # # print(checkpoint["hyper_parameters"])
    pl.seed_everything(cfg.seed)
    Extractor = ExtractorModel.load_from_checkpoint(checkpoint_path=checkpoint)
    # print(Extractor)
    val_subs = [1]
    train_subs = list(set(np.arange(cfg.data.n_subs)) - set(val_subs))
    # if len(val_subs) == 1:
    #     val_subs = list(val_subs) + train_subs
    print(train_subs)
    print(val_subs)
    data, onesub_label, n_samples_onesub, n_samples_sessions = load_processed_SEEDV_data(
            cfg.data.load_dir, cfg.data.fs, cfg.data.n_channs, cfg.data.timeLen,cfg.data.timeStep,cfg.data.n_session,cfg.data.n_subs,cfg.data.n_vids,cfg.data.n_class)
    data = data.reshape(cfg.data.n_subs, -1, data.shape[-2],data.shape[-1]) 
    data_train = data[list(train_subs)].reshape(-1, data.shape[-2], data.shape[-1])   # (train_subs*n_vids*30, 32, 250)
    data_val = data[list(val_subs)].reshape(-1, data.shape[-2], data.shape[-1])
    label_train = np.tile(onesub_label, len(train_subs))
    label_val = np.tile(onesub_label, len(val_subs))
    # trainset = SEEDV_Dataset(data_train, label_train)
    valset = SEEDV_Dataset(np.concatenate((data_val,data_train),0), np.concatenate((label_val,label_train),0))
    
    # train_sampler = TrainSampler_SEEDV(n_subs=len(train_subs), batch_size=cfg.data.n_vids,
    #                                     n_samples_session=n_samples_sessions, n_session=cfg.data.n_session, n_times=1)
    val_sampler = TrainSampler_SEEDV(n_subs=len(val_subs)+len(train_subs), batch_size=cfg.data.n_vids,
                                    n_samples_session=n_samples_sessions, n_session=cfg.data.n_session, n_times=1,if_val_loo=True)
    
    # train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, pin_memory=True, num_workers=8)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, pin_memory=True, num_workers=8)
    wandb_logger = WandbLogger(name=f'test1', project=cfg.log.proj_name)
    # dm = SEEDVDataModule(cfg.data.load_dir,cfg.data.save_dir, cfg.data.timeLen, cfg.data.timeStep, train_subs, val_subs, cfg.train.train_vids, cfg.train.val_vids, cfg.data.n_session, cfg.train.valid_method=='loo', cfg.train.num_workers)
    trainer = pl.Trainer(logger=wandb_logger, accelerator='gpu', devices=cfg.train.gpus)
    trainer.validate(Extractor, val_loader)

def path_test():
    print('path_test')
    # import sys
    # print(sys.path)
    
def grad_test():
    # 要在计算出现前（生成计算图之前）让requires_grad=True 才可以计算grad
    # detach()和detach_()生成的变量改requires_grad属性后还是可以计算grad
    import torch
    x = torch.tensor([1.],requires_grad=True)
    x_ = x.detach()
    
    y = torch.tensor([4.])
    x_.requires_grad = True
    out = x_*y
    
    
    
        
    print(x.requires_grad, y.requires_grad)
    print(x_.requires_grad)
    
    print(x.requires_grad, y.requires_grad)
    print(x_.requires_grad)
    

    out.sum().backward()
    print(x.grad)
    print(x_.grad)
    print(y.grad)

def test_array():
    # def add(a,b):
    #     a = a+b
    #     b = a-b
    # a = np.zeros((2,5))
    # # b = np.zeros((2,5))
    # b = np.ones_like(a)
    # c = np.concatenate((a,b),0)
    # c[0,1] = 1
    # print(a)
    # print(b)
    # print(c)
    a = np.ones((2,5))
    b = torch.FloatTensor(a)
    a[0,1] = 0
    print(b)
    
    # print(b)
    # a[0,1] = 1
    # print(b)
    # b[0,1] = 2
    # print(a)
    
def add():
    
    
    
    ave = np.mean([30.074073791503906, 15.037036895751953, 26.074073791503906, 18.370370864868164, 27.925926208496094, 
                   28.962963104248047, 32.66666793823242, 32.074073791503906, 31.629629135131836, 31.629629135131836, 
                   27.185184478759766, 25.33333396911621, 21.33333396911621, 33.62963104248047, 31.037036895751953])
    print(ave)

def norm_test():
    import torch.nn.functional as F
    a = torch.FloatTensor([3,4])
    print(F.normalize(a,dim=0))
    
def predict_test():
    a = np.zeros((2,3))
    b = np.array([-1,2]).reshape(2,1)
    print(a)
    print(a<b)

import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig
import os
import numpy as np  
@hydra.main(config_path='cfgs', config_name='config',version_base="1.3")
def fea_distribution(cfg: DictConfig):
    fold= 0
    save_path = os.path.join(cfg.ext_fea.save_dir,f'fold_{fold}_fea_'+cfg.ext_fea.mode+'.npy')
    data2 = np.load(save_path)
    fea = data2.reshape(16, -1, data2.shape[-1])
    val_subs = [0]
    train_subs = list(set(np.arange(16)) - set(val_subs))
    # Assuming fea, train_subs and val_subs are defined
    train_data = fea[train_subs[0]].reshape(-1, fea.shape[-1])
    val_data = fea[val_subs].reshape(-1, fea.shape[-1])
    print(train_data.shape, val_data.shape)

    # Plot histograms for each dimension
    for dim in range(fea.shape[-1]):
        plt.figure(figsize=(10, 6))
        overall_min = min(train_data[:, dim].min(), val_data[:, dim].min())
        overall_max = max(train_data[:, dim].max(), val_data[:, dim].max())
        plt.hist(train_data[:, dim], bins=30, range=(overall_min, overall_max), alpha=0.5, label='Train')
        plt.hist(val_data[:, dim], bins=30, range=(overall_min, overall_max), alpha=0.5, label='Validation')
        plt.title(f'Distribution for dimension {dim}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.legend(loc='upper right')
        plt.show()

def test_partial():
    import functools
    def add(a,b,c):
        print(a)
        print(b)
        print(c)
    add_ = functools.partial(add,a=1)
    print(add_(c=3,b=4))


if __name__ == "__main__":
    # path_test()
    # grad_test()
    # main2()
    # test_array()
    # add()
    # test_array()
    # norm_test()
    # fea_distribution()
    test_partial()