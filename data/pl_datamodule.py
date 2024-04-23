from .dataset import SEEDV_Dataset_new, PretrainSampler_SEEDV
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import os


class SEEDVDataModule(pl.LightningDataModule):
    def __init__(self, load_dir, save_dir, timeLen, timeStep, train_subs, val_subs, train_vids, val_vids, n_session=3, loo=False, num_workers=8) -> None:
        super().__init__()
        self.load_dir = load_dir
        self.save_dir = save_dir
        self.timeLen = timeLen
        self.timeStep = timeStep
        self.num_workers = num_workers
        self.train_subs = train_subs
        self.val_subs = val_subs
        self.loo = loo
        self.train_vids = train_vids
        self.val_vids = val_vids
        self.n_session = n_session
        self.sliced_data_dir = os.path.join(self.save_dir, f'sliced_len{self.timeLen}_step{self.timeStep}')
    
    def prepare_data(self):
        SEEDV_Dataset_new(self.load_dir,self.save_dir,self.timeLen,self.timeStep,sliced=False)
        print('prepare data finished!')
    
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.trainset = SEEDV_Dataset_new(self.load_dir,self.save_dir,self.timeLen,self.timeStep,train_subs=self.train_subs,mods='train',sliced=True)
            self.valset = SEEDV_Dataset_new(self.load_dir,self.save_dir,self.timeLen,self.timeStep,val_subs=self.val_subs,mods='val',sliced=True)
        if stage == 'validate':
            self.valset = SEEDV_Dataset_new(self.load_dir,self.save_dir,self.timeLen,self.timeStep,val_subs=self.val_subs,mods='val',sliced=True)
        
    def train_dataloader(self):
        self.n_samples_sessions = np.load(self.sliced_data_dir+'/metadata/n_samples_sessions.npy')
        train_sampler = PretrainSampler_SEEDV(n_subs=len(self.train_subs), batch_size=len(self.train_vids),
                                            n_samples_session=self.n_samples_sessions, n_session=self.n_session, n_times=1)
        return DataLoader(self.trainset, batch_sampler=train_sampler, pin_memory=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        self.n_samples_sessions = np.load(self.sliced_data_dir+'/metadata/n_samples_sessions.npy')
        val_sampler = PretrainSampler_SEEDV(len(self.val_subs), batch_size=len(self.val_vids),
                                        n_samples_session=self.n_samples_sessions, n_session=self.n_session, n_times=1,if_val_loo=self.loo)
        return DataLoader(self.valset, batch_sampler=val_sampler, pin_memory=True, num_workers=self.num_workers)
    
if __name__ == '__main__':
    pass
    