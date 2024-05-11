from torch.utils.data import Dataset, DataLoader, Sampler
import torch
from .io_utils import load_processed_SEEDV_NEW_data, save_sliced_data
import os
import numpy as np
import random
from functools import partial



class FACED_Dataset(Dataset):
    def __init__(self, data, label):
        self.data = torch.FloatTensor(data) # n_samples * n_features
        self.label = torch.from_numpy(label)
        # self.sub_label = torch.from_numpy(sub_label)
    def __len__(self):
        return len(self.label)
    def __getitem__(self, idx):
        one_seq = self.data[idx].reshape(1,self.data.shape[-2],self.data.shape[-1])      # 32*(250)->1*32*250  2d conv  c*h*w
        one_label = self.label[idx]
        # one_sub_label = self.sub_label[idx]
        return one_seq, one_label

class SEEDV_Dataset(Dataset):
    def __init__(self, data, label):
        self.data = torch.FloatTensor(data) # n_samples * n_features
        self.label = torch.from_numpy(label)
        # self.sub_label = torch.from_numpy(sub_label)
    def __len__(self):
        return len(self.label)
    def __getitem__(self, idx):
        one_seq = self.data[idx].reshape(1,self.data.shape[-2],self.data.shape[-1])      # 32*(250)->1*32*250  2d conv  c*h*w
        one_label = self.label[idx]
        # one_sub_label = self.sub_label[idx]
        return one_seq, one_label
    
class SEEDV_Dataset_new(Dataset):
    def __init__(self, load_dir, save_dir, timeLen, timeStep, train_subs=None, val_subs=None, sliced=True, mods='train', n_session=3, fs=125, n_chans=60, n_subs=16, n_vids = 15, n_class=5):
        self.load_dir = load_dir
        self.save_dir = save_dir
        self.n_subs = n_subs
        self.timeLen = timeLen
        self.timeStep = timeStep
        self.train_subs = train_subs
        self.val_subs = val_subs
        self.mods = mods
        self.sliced_data_dir = os.path.join(self.save_dir, f'sliced_len{self.timeLen}_step{self.timeStep}')
        self.load_processed_SEEDV_NEW_data = partial(load_processed_SEEDV_NEW_data, dir=self.load_dir, 
                                                     timeLen=self.timeLen, timeStep=self.timeStep)
        self.save_sliced_data = partial(save_sliced_data, sliced_data_dir = self.sliced_data_dir)
        if not sliced:
            if not os.path.exists(self.sliced_data_dir+'/saved.npy'):
                print('slicing processed dataset')
                data, onesub_labels, n_samples_onesub, n_samples_sessions = self.load_processed_SEEDV_NEW_data(
                    fs=fs, n_chans=n_chans, n_session=n_session, n_subs=n_subs, n_vids=n_vids, n_class=n_class)
                self.save_sliced_data(data=data, onesub_labels=onesub_labels, n_samples_onesub=n_samples_onesub, n_samples_sessions=n_samples_sessions)
            else:
                print('sliced data exist!')
        
        self.onesub_label = torch.from_numpy(np.load(os.path.join(self.sliced_data_dir, 'metadata', 'onesub_labels.npy')))
        self.labels = self.onesub_label.repeat(self.n_subs)
        self.onesubLen = len(self.onesub_label)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if self.mods == 'train':
            if self.train_subs is not None:
                idx = self.train_subs[idx//self.onesubLen]*self.onesubLen + idx%self.onesubLen
        elif self.mods == 'val':
            if self.val_subs is not None:
                idx = self.val_subs[idx//self.onesubLen]*self.onesubLen + idx%self.onesubLen
        one_seq = np.load(os.path.join(self.sliced_data_dir, 'data', f'data_sample_{idx}.npy'))
        one_seq = torch.FloatTensor(one_seq.reshape(1,one_seq.shape[-2],one_seq.shape[-1]))    # 32*(250)->1*32*250  2d conv  c*h*w
        one_label = self.labels[idx]
        # one_sub_label = self.sub_label[idx]
        return one_seq, one_label
    
    # def save_sliced_data(self, data, onesub_labels, n_samples_onesub, n_samples_sessions):
    #     if not os.path.exists(self.sliced_data_dir+'/metadata'):
    #         os.makedirs(self.sliced_data_dir+'/metadata')
    #     if not os.path.exists(self.sliced_data_dir+'/data'):
    #         os.makedirs(self.sliced_data_dir+'/data')
    #     np.save(self.sliced_data_dir+'/metadata/onesub_labels.npy', onesub_labels)
    #     np.save(self.sliced_data_dir+'/metadata/n_samples_onesub.npy', n_samples_onesub)
    #     np.save(self.sliced_data_dir+'/metadata/n_samples_sessions.npy', n_samples_sessions)
    #     for sample in range(data.shape[0]):
    #         np.save(self.sliced_data_dir+f'/data/data_sample_{sample}.npy', data[sample])
    #     np.save(self.sliced_data_dir+'/saved.npy', [True])
    #     print('save sliced data finished!')
                
    # def load_processed_SEEDV_data(self, n_session=3, fs=250, n_chans=62, n_subs=16, n_vids = 15, n_class=5):
    #     # input data shape(onesub_onesession):(channels,tot_time) tot_time = sum(eachvids_n_points) 
    #     # output : (subs*sum(n_samples_onesub))*channals*time
    #     #           (15*(sum(n_samples_onesub)))*62*point_len(1250)
        
    #     dir = self.load_dir
    #     # print(dir)
    #     timeLen = self.timeLen
    #     timeStep = self.timeStep
        
    #     list_files = os.listdir(dir)
    #     list_files.sort(key= lambda x:int(x[:-4]))
    #     # print(list_files)

    #     points_len = int(timeLen*fs)
    #     points_step = int(timeStep*fs)


    #     n_samples_onesub = []
    #     for i in range(n_session):
    #         fn = list_files[i]
    #         file_path = os.path.join(dir,fn)
    #         onesubsession_data = sio.loadmat(file_path)  
    #         n_points = np.squeeze(onesubsession_data['n_points']).astype(int)
    #         n_samples_onesubsession = ((n_points-points_len)//points_step+1).astype(int)
    #         n_samples_onesub = n_samples_onesub + list(n_samples_onesubsession)

    #     n_samples_sum_onesub = np.sum(n_samples_onesub)


    #     data = np.empty((n_subs*n_samples_sum_onesub,n_chans,points_len),float)

    #     s = np.arange(n_session)
    #     # n_samples_onesub = []
    #     cnt = 0
    #     for idx,fn in enumerate(list_files):
    #         file_path = os.path.join(dir,fn)
    #         # print(fn)
    #         onesubsession_data = sio.loadmat(file_path)     #keys: data,n_points
    #         EEG_data = onesubsession_data['data']   #(channels,tot_n_points)  (62,tot_n_points)
    #         thr = 30 * np.median(np.abs(EEG_data))
    #         EEG_data = (EEG_data - np.mean(EEG_data[EEG_data<thr])) / np.std(EEG_data[EEG_data<thr])
    #         n_points = np.squeeze(onesubsession_data['n_points']).astype(int)
    #         # print(EEG_data.shape)
    #         n_points_cum = np.concatenate((np.array([0]),np.cumsum(n_points)))
    #         n_samples_onesubsession = ((n_points-points_len)//points_step+1).astype(int)

    #         # if idx < n_session:
    #         #     if idx == s[idx]:
    #         #         n_samples_onesub = n_samples_onesub + list(n_samples_onesubsession)
    #         for vid in range(n_vids):
    #             # print('vid:',vid)
    #             for i in range(n_samples_onesubsession[vid]):
    #                 # print('sample:',i)

    #                 data[cnt] = EEG_data[:,n_points_cum[vid]+i*points_step:n_points_cum[vid]+i*points_step+points_len]
    #                 cnt+=1

    #                 # 拼接速度会越来越慢
    #                 # temp = temp.reshape(1,temp.shape[0],temp.shape[1])
    #                 # start_time = time.time()
    #                 # data = np.concatenate((data,temp),0)
    #                 # end_time = time.time()
    #                 # print(end_time - start_time)
    #     # print(cnt)

    #     n_samples_onesub = np.array(n_samples_onesub)
    #     n_samples_sessions = n_samples_onesub.reshape(n_session,-1)
    #     label = [4, 1, 3, 2, 0] * 3 + [2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0] * 2
    #     onesub_label = []
    #     for i in range(len(label)):
    #         onesub_label = onesub_label + [label[i]]*n_samples_onesub[i]   

    #     return data, np.array(onesub_label), n_samples_onesub, n_samples_sessions

    # def load_processed_SEEDV_NEW_data(self, n_session=3, fs=125, n_chans=60, n_subs=16, n_vids = 15, n_class=5):
    #     # input data shape(onesub_onesession):(channels,tot_time) tot_time = sum(eachvids_n_points) 
    #     # *input data shape（onesub_3session):(channels,tot_time)
    #     # output : (subs*sum(n_samples_onesub))*channals*time
    #     #           (16*(sum(n_samples_onesub)))*62*point_len(1250)
        
    #     dir = self.load_dir
    #     # print(dir)
    #     timeLen = self.timeLen
    #     timeStep = self.timeStep
        
    #     list_files = os.listdir(dir)
    #     print("list_files:", list_files)
    #     # list_files.sort(key= lambda x:int(x[:-4]))
    #     list_files = sorted(list_files, key=lambda x: int(re.search(r'\d+', x).group()))

    #     # print(list_files)

    #     points_len = int(timeLen*fs)
    #     points_step = int(timeStep*fs)

    #     # 3 session in all change delete the loop
    #     # n_samples_onesub = []
    #     # for fn in list_files:
    #     file_path = os.path.join(dir,list_files[0])
    #     onesub_data = sio.loadmat(file_path)  
    #     n_time = np.squeeze(onesub_data['merged_n_samples_one']).astype(int)
    #     n_points = np.array(n_time) * fs
    #     # n_points = np.squeeze(onesubsession_data['n_points']).astype(int)
    #     n_samples_onesub = ((n_points-points_len)//points_step+1).astype(int)
    #     # n_samples_onesub = n_samples_onesub + list(n_samples_onesubsession)

    #     n_samples_sum_onesub = np.sum(n_samples_onesub)


    #     data = np.empty((n_subs*n_samples_sum_onesub,n_chans,points_len),float)


    #     cnt = 0
    #     for idx,fn in enumerate(list_files):
    #         file_path = os.path.join(dir,fn)
    #         # print(fn)
    #         onesub_data = sio.loadmat(file_path)     #keys: data,n_points
    #         EEG_data = onesub_data['merged_data_all_cleaned']   #(channels,tot_n_points_3session)  (60,tot_n_points_3session)
    #         thr = 30 * np.median(np.abs(EEG_data))
    #         EEG_data = (EEG_data - np.mean(EEG_data[EEG_data<thr])) / np.std(EEG_data[EEG_data<thr])
    #         # n_points = np.squeeze(onesubsession_data['n_points']).astype(int)
    #         # print(EEG_data.shape)
    #         n_points_cum = np.concatenate((np.array([0]),np.cumsum(n_points)))
    #         # n_samples_onesub = ((n_points-points_len)//points_step+1).astype(int)
            
    #         # if idx < n_session:
    #         #     if idx == s[idx]:
    #         #         n_samples_onesub = n_samples_onesub + list(n_samples_onesubsession)
            
    #         n_vids = 45
    #         for vid in range(n_vids):
    #             print('vid:',vid)
    #             for i in range(n_samples_onesub[vid]):
    #                 # print('sample:',i)

    #                 data[cnt] = EEG_data[:,n_points_cum[vid]+i*points_step:n_points_cum[vid]+i*points_step+points_len]
    #                 cnt+=1

    #                 # 拼接速度会越来越慢
    #                 # temp = temp.reshape(1,temp.shape[0],temp.shape[1])
    #                 # start_time = time.time()
    #                 # data = np.concatenate((data,temp),0)
    #                 # end_time = time.time()
    #                 # print(end_time - start_time)
    #     # print(cnt)
    #     print("[Test]shape before reshape:",data.shape)

    #     n_samples_onesub = np.array(n_samples_onesub)
    #     n_samples_sessions = n_samples_onesub.reshape(n_session,-1)
    #     label = [4, 1, 3, 2, 0] * 3 + [2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0] * 2
    #     onesub_label = []
    #     for i in range(len(label)):
    #         onesub_label = onesub_label + [label[i]]*n_samples_onesub[i]   

    #     return data, np.array(onesub_label), n_samples_onesub, n_samples_sessions


    
class PDataset(Dataset):
    def __init__(self, data, label):
        self.data = torch.FloatTensor(data) # n_samples * n_features
        self.label = torch.from_numpy(label)
    def __len__(self):
        return len(self.label)
    def __getitem__(self, idx):
        one_seq = self.data[idx]
        one_label = self.label[idx]
        return one_seq, one_label


class TrainSampler_FACED():
    def __init__(self, n_subs, batch_size, n_samples, n_session=1, n_times=1):
        # input
        # n_per: 一个sub的采样数总和   n_samples_inonesub = n_vids*n_samples_inonevid  
        # n_sub: 数据集被试数量，包括同一个被试的不同session 
        # batch_size： 对比学习时组成的一个sub pair的一次采样中，采样视频对的个数，一般取n_vids的k倍，也即一个视频取k个对，不重复的取，见n_samples_per_trial
        # n_samples_cum：  累计的sample数，
        # n_session: 一个sub当中有几个session
        # n_samples_per_trial：int(batch_size / len(n_samples))  一个sub pair的一次采样中，采样视频对的个数
        # sub_pairs：组成的sub对，需要不同的sub，同一个sub的不同session不可以组对
        # n_times：一个sub pair的采样次数，不同次采样可能采到相同的视频对
        self.n_per = int(np.sum(n_samples))
        self.n_subs = n_subs
        self.batch_size = batch_size
        self.n_samples_cum = np.concatenate((np.array([0]), np.cumsum(n_samples)))
        self.n_samples_per_trial = int(batch_size / len(n_samples))
        self.sub_pairs = []
        for i in range(self.n_subs):
            # j = i
            for j in range(i+n_session, self.n_subs, n_session):
                self.sub_pairs.append([i, j])
        random.shuffle(self.sub_pairs)
        # 采样次数
        self.n_times = n_times

    def __len__(self):
        return self.n_times * len(self.sub_pairs)

    def __iter__(self):
        for s in range(len(self.sub_pairs)):
            for t in range(self.n_times):
                [sub1, sub2] = self.sub_pairs[s]
                # print(sub1, sub2)

                ind_abs = np.zeros(0)
                if self.batch_size < len(self.n_samples_cum)-1:
                    sel_vids = np.random.choice(np.arange(len(self.n_samples_cum)-1), self.batch_size)
                    for i in sel_vids:
                        ind_one = np.random.choice(np.arange(self.n_samples_cum[i], self.n_samples_cum[i+1]), 1, replace=False)
                        ind_abs = np.concatenate((ind_abs, ind_one))
                else:
                    for i in range(len(self.n_samples_cum)-2):
                        ind_one = np.random.choice(np.arange(self.n_samples_cum[i], self.n_samples_cum[i+1]),
                                                   self.n_samples_per_trial, replace=False)
                        ind_abs = np.concatenate((ind_abs, ind_one))

                    i = len(self.n_samples_cum) - 2
                    ind_one = np.random.choice(np.arange(self.n_samples_cum[i], self.n_samples_cum[i + 1]),
                                               int(self.batch_size - len(ind_abs)), replace=False)
                    ind_abs = np.concatenate((ind_abs, ind_one))
                    # print('ind abs length', len(ind_abs))

                assert len(ind_abs) == self.batch_size
                # ind_abs = np.arange(self.batch_size)

                # print(ind_abs)
                ind_this1 = ind_abs + self.n_per*sub1
                ind_this2 = ind_abs + self.n_per*sub2

                batch = torch.LongTensor(np.concatenate((ind_this1, ind_this2)))
                yield batch


class TrainSampler_SEEDV():
    def __init__(self, n_subs, batch_size, n_samples_session, n_session=1, n_times=1, if_val_loo=False):
        # input
        # n_per_session: 一个sub的采样数总和   n_samples_inonesub = n_vids*n_samples_inonevid  session维度
        # n_sub: 数据集被试数量，不包括同一个被试的不同session 
        # batch_size： 对比学习时组成的一个sub pair的一次采样中，采样视频对的个数，一般取n_vids的k倍，也即一个视频取k个对，不重复的取，见n_samples_per_trial
        # n_samples_session （n_session,n_vids）
        # n_samples_cum_session  （n_session,n_vids+1） 累计的sample数，
        # n_session: 一个sub当中有几个session
        # n_samples_per_trial：int(batch_size / len(n_samples))  一个sub pair的一次采样中，采样视频对的个数
        # subsession_pairs：组成的subsession对，需要不同的sub，相同的session
        # n_times：一个sub pair的采样次数，不同次采样可能采到相同的视频对

        
        self.n_per_session = np.sum(n_samples_session,1).astype(int)
        self.n_per_session_cum = np.concatenate((np.array([0]), np.cumsum(self.n_per_session)))
        self.n_subs = n_subs
        self.batch_size = batch_size
        self.n_samples_cum_session = np.concatenate((np.zeros((n_session,1)), np.cumsum(n_samples_session,1)),1)
        self.n_samples_per_trial = int(batch_size / n_samples_session.shape[1])
        self.subsession_pairs = []
        self.n_session = n_session
        if if_val_loo:
            self.n_pairsubs = 1
        else:
            self.n_pairsubs = self.n_subs
        for i in range(self.n_pairsubs*self.n_session):
            # j = i
            for j in range(i+n_session, self.n_subs*self.n_session, n_session):
                self.subsession_pairs.append([i, j])
        random.shuffle(self.subsession_pairs)
        # 采样次数
        self.n_times = n_times

    def __len__(self):   #n_batch
        return self.n_times * len(self.subsession_pairs)

    def __iter__(self):
        for s in range(len(self.subsession_pairs)):
            for t in range(self.n_times):
                [subsession1, subsession2] = self.subsession_pairs[s]
                cur_session = int(subsession1 % self.n_session)
                cur_sub1 = int(subsession1 // self.n_session)
                cur_sub2 = int(subsession2 // self.n_session)

                ind_abs = np.zeros(0)
                if self.batch_size < len(self.n_samples_cum_session[cur_session])-1:
                    sel_vids = np.random.choice(np.arange(len(self.n_samples_cum_session[cur_session])-1), self.batch_size)
                    for i in sel_vids:
                        ind_one = np.random.choice(np.arange(self.n_samples_cum_session[cur_session][i], self.n_samples_cum_session[cur_session][i+1]), 1, replace=False)
                        ind_abs = np.concatenate((ind_abs, ind_one))
                else:
                    for i in range(len(self.n_samples_cum_session[cur_session])-2):
                        ind_one = np.random.choice(np.arange(self.n_samples_cum_session[cur_session][i], self.n_samples_cum_session[cur_session][i+1]),
                                                   self.n_samples_per_trial, replace=False)
                        ind_abs = np.concatenate((ind_abs, ind_one))

                    i = len(self.n_samples_cum_session[cur_session]) - 2
                    ind_one = np.random.choice(np.arange(self.n_samples_cum_session[cur_session][i], self.n_samples_cum_session[cur_session][i + 1]),
                                               int(self.batch_size - len(ind_abs)), replace=False)
                    ind_abs = np.concatenate((ind_abs, ind_one))
                    # print('ind abs length', len(ind_abs))

                assert len(ind_abs) == self.batch_size
                # ind_abs = np.arange(self.batch_size)

                # print(ind_abs)
                # print(cur_sub1)
                # print(cur_sub2)
                # print(cur_session)
                # print(self.n_per_session)
                # print(self.n_per_session_cum)
                # print(self.n_samples_cum_session)
                ind_this1 = ind_abs + np.sum(self.n_per_session)*cur_sub1 + self.n_per_session_cum[cur_session]
                ind_this2 = ind_abs + np.sum(self.n_per_session)*cur_sub2 + self.n_per_session_cum[cur_session]

                batch = torch.LongTensor(np.concatenate((ind_this1, ind_this2)))
                # print(batch)
                yield batch


class PretrainSampler_SEEDV():
    def __init__(self, n_subs, batch_size, n_samples_session, n_session=1, n_times=1, if_val_loo=False):
        # input
        # n_per_session: 一个sub的采样数总和   n_samples_inonesub = n_vids*n_samples_inonevid  session维度
        # n_sub: 数据集被试数量，不包括同一个被试的不同session 
        # batch_size： 对比学习时组成的一个sub pair的一次采样中，采样视频对的个数，一般取n_vids的k倍，也即一个视频取k个对，不重复的取，见n_samples_per_trial
        # n_samples_session （n_session,n_vids）
        # n_samples_cum_session  （n_session,n_vids+1） 累计的sample数，
        # n_session: 一个sub当中有几个session
        # n_samples_per_trial：int(batch_size / len(n_samples))  一个sub pair的一次采样中，采样视频对的个数
        # subsession_pairs：组成的subsession对，需要不同的sub，相同的session
        # n_times：一个sub pair的采样次数，不同次采样可能采到相同的视频对

        
        self.n_per_session = np.sum(n_samples_session,1).astype(int)
        self.n_per_session_cum = np.concatenate((np.array([0]), np.cumsum(self.n_per_session)))
        self.n_subs = n_subs
        self.batch_size = batch_size
        self.n_samples_cum_session = np.concatenate((np.zeros((n_session,1)), np.cumsum(n_samples_session,1)),1)
        self.n_samples_per_trial = int(batch_size / n_samples_session.shape[1])
        self.subsession_pairs = []
        self.n_session = n_session
        if if_val_loo:
            self.n_pairsubs = 1
        else:
            self.n_pairsubs = self.n_subs
        for i in range(self.n_pairsubs*self.n_session):
            # j = i
            for j in range(i+n_session, self.n_subs*self.n_session, n_session):
                self.subsession_pairs.append([i, j])
        random.shuffle(self.subsession_pairs)
        # 采样次数
        self.n_times = n_times

    def __len__(self):   #n_batch
        return self.n_times * len(self.subsession_pairs)

    def __iter__(self):
        for s in range(len(self.subsession_pairs)):
            for t in range(self.n_times):
                [subsession1, subsession2] = self.subsession_pairs[s]
                cur_session = int(subsession1 % self.n_session)
                cur_sub1 = int(subsession1 // self.n_session)
                cur_sub2 = int(subsession2 // self.n_session)

                ind_abs = np.zeros(0)
                if self.batch_size < len(self.n_samples_cum_session[cur_session])-1:
                    sel_vids = np.random.choice(np.arange(len(self.n_samples_cum_session[cur_session])-1), self.batch_size)
                    for i in sel_vids:
                        ind_one = np.random.choice(np.arange(self.n_samples_cum_session[cur_session][i], self.n_samples_cum_session[cur_session][i+1]), 1, replace=False)
                        ind_abs = np.concatenate((ind_abs, ind_one))
                else:
                    for i in range(len(self.n_samples_cum_session[cur_session])-2):
                        ind_one = np.random.choice(np.arange(self.n_samples_cum_session[cur_session][i], self.n_samples_cum_session[cur_session][i+1]),
                                                   self.n_samples_per_trial, replace=False)
                        ind_abs = np.concatenate((ind_abs, ind_one))

                    i = len(self.n_samples_cum_session[cur_session]) - 2
                    ind_one = np.random.choice(np.arange(self.n_samples_cum_session[cur_session][i], self.n_samples_cum_session[cur_session][i + 1]),
                                               int(self.batch_size - len(ind_abs)), replace=False)
                    ind_abs = np.concatenate((ind_abs, ind_one))
                    # print('ind abs length', len(ind_abs))

                assert len(ind_abs) == self.batch_size
                # ind_abs = np.arange(self.batch_size)

                # print(ind_abs)
                # print(cur_sub1)
                # print(cur_sub2)
                # print(cur_session)
                # print(self.n_per_session)
                # print(self.n_per_session_cum)
                # print(self.n_samples_cum_session)
                ind_this1 = ind_abs + np.sum(self.n_per_session)*cur_sub1 + self.n_per_session_cum[cur_session]
                ind_this2 = ind_abs + np.sum(self.n_per_session)*cur_sub2 + self.n_per_session_cum[cur_session]

                batch = torch.LongTensor(np.concatenate((ind_this1, ind_this2)))
                # print(batch)
                yield batch



class vid_sampler(Sampler):
    def __init__(self,n_sub,n_vid,n_sample) -> None:
        self.n_sub = n_sub
        self.n_sample = n_sample
        self.n_vid = n_vid
        # self.data_source = data_source

    def __len__(self):
        # print(len(self.data_source))
        return self.n_sub*self.n_sample*self.n_vid
        

    def __iter__(self):
        # random_choice = torch.randperm(self.n_vid*self.n_sub).numpy()*self.n_sample
        random_choice = np.random.permutation(np.arange(self.n_sub*self.n_vid))*(self.n_sample)
        sample_array = []
        for idx in random_choice:
            temp = np.arange(idx,idx+self.n_sample).tolist()
            sample_array = sample_array + temp
        # print(sample_array)
        # print(len(sample_array))
        return iter(sample_array)

class sub_sampler(Sampler):
    def __init__(self,n_sub,n_vid,n_sample) -> None:
        self.n_sub = n_sub
        self.n_sample = n_sample
        self.n_vid = n_vid
        # self.data_source = data_source

    def __len__(self):
        # print(len(self.data_source))
        return self.n_sub*self.n_sample*self.n_vid
        

    def __iter__(self):
        # random_choice = torch.randperm(self.n_sub)*self.n_sample
        random_choice = np.random.permutation(np.arange(self.n_sub))*(self.n_sample*self.n_vid)

        sample_array = []
        for idx in random_choice:
            temp = np.arange(idx,idx+self.n_sample*self.n_vid).tolist()
            sample_array = sample_array + temp
        # print(sample_array)
        # print(len(sample_array))
        return iter(sample_array)


if __name__ == '__main__':
    pass


    
