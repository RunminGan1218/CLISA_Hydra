from torch.utils.data import Dataset, DataLoader, Sampler
import torch
import pickle
import os
import numpy as np
import random
import scipy.io as sio
import re



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
        if not sliced:
            if not os.path.exists(self.sliced_data_dir+'/saved.npy'):
                print('slicing processed dataset')
                data, onesub_label, n_samples_onesub, n_samples_sessions = self.load_processed_SEEDV_NEW_data(n_session, fs, n_chans, n_subs, n_vids, n_class)
                self.save_sliced_data(data, onesub_label, n_samples_onesub, n_samples_sessions)
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
    
    def save_sliced_data(self, data, onesub_labels, n_samples_onesub, n_samples_sessions):
        if not os.path.exists(self.sliced_data_dir+'/metadata'):
            os.makedirs(self.sliced_data_dir+'/metadata')
        if not os.path.exists(self.sliced_data_dir+'/data'):
            os.makedirs(self.sliced_data_dir+'/data')
        np.save(self.sliced_data_dir+'/metadata/onesub_labels.npy', onesub_labels)
        np.save(self.sliced_data_dir+'/metadata/n_samples_onesub.npy', n_samples_onesub)
        np.save(self.sliced_data_dir+'/metadata/n_samples_sessions.npy', n_samples_sessions)
        for sample in range(data.shape[0]):
            np.save(self.sliced_data_dir+f'/data/data_sample_{sample}.npy', data[sample])
        np.save(self.sliced_data_dir+'/saved.npy', [True])
        print('save sliced data finished!')
                
    def load_processed_SEEDV_data(self, n_session=3, fs=250, n_chans=62, n_subs=16, n_vids = 15, n_class=5):
        # input data shape(onesub_onesession):(channels,tot_time) tot_time = sum(eachvids_n_points) 
        # output : (subs*sum(n_samples_onesub))*channals*time
        #           (15*(sum(n_samples_onesub)))*62*point_len(1250)
        
        dir = self.load_dir
        # print(dir)
        timeLen = self.timeLen
        timeStep = self.timeStep
        
        list_files = os.listdir(dir)
        list_files.sort(key= lambda x:int(x[:-4]))
        # print(list_files)

        points_len = int(timeLen*fs)
        points_step = int(timeStep*fs)


        n_samples_onesub = []
        for i in range(n_session):
            fn = list_files[i]
            file_path = os.path.join(dir,fn)
            onesubsession_data = sio.loadmat(file_path)  
            n_points = np.squeeze(onesubsession_data['n_points']).astype(int)
            n_samples_onesubsession = ((n_points-points_len)//points_step+1).astype(int)
            n_samples_onesub = n_samples_onesub + list(n_samples_onesubsession)

        n_samples_sum_onesub = np.sum(n_samples_onesub)


        data = np.empty((n_subs*n_samples_sum_onesub,n_chans,points_len),float)

        s = np.arange(n_session)
        # n_samples_onesub = []
        cnt = 0
        for idx,fn in enumerate(list_files):
            file_path = os.path.join(dir,fn)
            # print(fn)
            onesubsession_data = sio.loadmat(file_path)     #keys: data,n_points
            EEG_data = onesubsession_data['data']   #(channels,tot_n_points)  (62,tot_n_points)
            thr = 30 * np.median(np.abs(EEG_data))
            EEG_data = (EEG_data - np.mean(EEG_data[EEG_data<thr])) / np.std(EEG_data[EEG_data<thr])
            n_points = np.squeeze(onesubsession_data['n_points']).astype(int)
            # print(EEG_data.shape)
            n_points_cum = np.concatenate((np.array([0]),np.cumsum(n_points)))
            n_samples_onesubsession = ((n_points-points_len)//points_step+1).astype(int)

            # if idx < n_session:
            #     if idx == s[idx]:
            #         n_samples_onesub = n_samples_onesub + list(n_samples_onesubsession)
            for vid in range(n_vids):
                # print('vid:',vid)
                for i in range(n_samples_onesubsession[vid]):
                    # print('sample:',i)

                    data[cnt] = EEG_data[:,n_points_cum[vid]+i*points_step:n_points_cum[vid]+i*points_step+points_len]
                    cnt+=1

                    # 拼接速度会越来越慢
                    # temp = temp.reshape(1,temp.shape[0],temp.shape[1])
                    # start_time = time.time()
                    # data = np.concatenate((data,temp),0)
                    # end_time = time.time()
                    # print(end_time - start_time)
        # print(cnt)

        n_samples_onesub = np.array(n_samples_onesub)
        n_samples_sessions = n_samples_onesub.reshape(n_session,-1)
        label = [4, 1, 3, 2, 0] * 3 + [2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0] * 2
        onesub_label = []
        for i in range(len(label)):
            onesub_label = onesub_label + [label[i]]*n_samples_onesub[i]   

        return data, np.array(onesub_label), n_samples_onesub, n_samples_sessions

    def load_processed_SEEDV_NEW_data(self, n_session=3, fs=125, n_chans=60, n_subs=16, n_vids = 15, n_class=5):
        # input data shape(onesub_onesession):(channels,tot_time) tot_time = sum(eachvids_n_points) 
        # *input data shape（onesub_3session):(channels,tot_time)
        # output : (subs*sum(n_samples_onesub))*channals*time
        #           (16*(sum(n_samples_onesub)))*62*point_len(1250)
        
        dir = self.load_dir
        # print(dir)
        timeLen = self.timeLen
        timeStep = self.timeStep
        
        list_files = os.listdir(dir)
        print("list_files:", list_files)
        # list_files.sort(key= lambda x:int(x[:-4]))
        list_files = sorted(list_files, key=lambda x: int(re.search(r'\d+', x).group()))

        # print(list_files)

        points_len = int(timeLen*fs)
        points_step = int(timeStep*fs)


        n_samples_onesub = []
        for fn in list_files:
            file_path = os.path.join(dir,fn)
            onesub_data = sio.loadmat(file_path)  
            n_time = np.squeeze(onesub_data['merged_n_samples_one']).astype(int)
            n_points = np.array(n_time) * fs
            # n_points = np.squeeze(onesubsession_data['n_points']).astype(int)
            n_samples_onesub = ((n_points-points_len)//points_step+1).astype(int)
            # n_samples_onesub = n_samples_onesub + list(n_samples_onesubsession)

        n_samples_sum_onesub = np.sum(n_samples_onesub)


        data = np.empty((n_subs*n_samples_sum_onesub,n_chans,points_len),float)

        s = np.arange(n_session)
        # n_samples_onesub = []
        cnt = 0
        for idx,fn in enumerate(list_files):
            file_path = os.path.join(dir,fn)
            # print(fn)
            onesubsession_data = sio.loadmat(file_path)     #keys: data,n_points
            EEG_data = onesubsession_data['merged_data_all_cleaned']   #(channels,tot_n_points)  (62,tot_n_points)
            thr = 30 * np.median(np.abs(EEG_data))
            EEG_data = (EEG_data - np.mean(EEG_data[EEG_data<thr])) / np.std(EEG_data[EEG_data<thr])
            # n_points = np.squeeze(onesubsession_data['n_points']).astype(int)
            # print(EEG_data.shape)
            n_points_cum = np.concatenate((np.array([0]),np.cumsum(n_points)))
            n_samples_onesub = ((n_points-points_len)//points_step+1).astype(int)
            
            # if idx < n_session:
            #     if idx == s[idx]:
            #         n_samples_onesub = n_samples_onesub + list(n_samples_onesubsession)
            
            n_vids = 45
            for vid in range(n_vids):
                print('vid:',vid)
                for i in range(n_samples_onesub[vid]):
                    # print('sample:',i)

                    data[cnt] = EEG_data[:,n_points_cum[vid]+i*points_step:n_points_cum[vid]+i*points_step+points_len]
                    cnt+=1

                    # 拼接速度会越来越慢
                    # temp = temp.reshape(1,temp.shape[0],temp.shape[1])
                    # start_time = time.time()
                    # data = np.concatenate((data,temp),0)
                    # end_time = time.time()
                    # print(end_time - start_time)
        # print(cnt)
        print("[Test]shape before reshape:",data.shape)

        n_samples_onesub = np.array(n_samples_onesub)
        n_samples_sessions = n_samples_onesub.reshape(n_session,-1)
        label = [4, 1, 3, 2, 0] * 3 + [2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0] * 2
        onesub_label = []
        for i in range(len(label)):
            onesub_label = onesub_label + [label[i]]*n_samples_onesub[i]   

        return data, np.array(onesub_label), n_samples_onesub, n_samples_sessions


    
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
    
    

def load_processed_FACED_data(dir, fs, n_chans, t, timeLen,timeStep, n_class):
    # input data shape(onesub):(vid,channel,time)
    # output : subs*(slices*vids)*channals*time
    #           123*(28*30)*32*(250)

    list_files = os.listdir(dir)
    n_samples = int((t-timeLen)/timeStep)+1
    

    if n_class == 2:
        vid_sel = list(range(12))
        vid_sel.extend(list(range(16,28)))
        # data = data[:, vid_sel, :, :] # sub, vid, n_channs, n_points
        n_vids = 24
    elif n_class == 9:
        vid_sel = list(range(28))
        n_vids = 28
    data = np.empty((len(list_files),n_vids*n_samples,n_chans,fs*timeLen),float)

    for idx,fn in enumerate(list_files):
        file_path = os.path.join(dir,fn)
        
        with open(file_path, 'rb') as fo:     # 读取pkl文件数据
            onesub_data = pickle.load(fo, encoding='bytes')
            for k, vid in enumerate(vid_sel):
                for i in range(n_samples):
                    data[idx,k*n_samples+i] = onesub_data[vid,:,int(i*fs*timeStep):int(i*fs*timeStep+timeLen*fs)]

    if n_class == 2:
        label = [0] * 12
        label.extend([1] * 12)
    elif n_class == 9:
        label = [0] * 3
        for i in range(1,4):
            label.extend([i] * 3)
        label.extend([4] * 4)
        for i in range(5,9):
            label.extend([i] * 3)
    
    label_repeat = []
    for i in range(len(label)):
        label_repeat = label_repeat + [label[i]]*n_samples

    return data, np.array(label_repeat), n_samples

def load_processed_SEEDV_data(dir, fs, n_chans, timeLen,timeStep, n_session, n_subs=16, n_vids = 15, n_class=5):
    # input data shape(onesub_onesession):(channels,tot_time) tot_time = sum(eachvids_n_points) 
    # output : (subs*sum(n_samples_onesub))*channals*time
    #           (15*(sum(n_samples_onesub)))*62*point_len(1250)

    list_files = os.listdir(dir)
    list_files.sort(key= lambda x:int(x[:-4]))
    # print(list_files)

    points_len = int(timeLen*fs)
    points_step = int(timeStep*fs)


    n_samples_onesub = []
    for i in range(n_session):
        fn = list_files[i]
        file_path = os.path.join(dir,fn)
        onesubsession_data = sio.loadmat(file_path)  
        n_points = np.squeeze(onesubsession_data['n_points']).astype(int)
        n_samples_onesubsession = ((n_points-points_len)//points_step+1).astype(int)
        n_samples_onesub = n_samples_onesub + list(n_samples_onesubsession)

    n_samples_sum_onesub = np.sum(n_samples_onesub)


    data = np.empty((n_subs*n_samples_sum_onesub,n_chans,points_len),float)

    s = np.arange(n_session)
    # n_samples_onesub = []
    cnt = 0
    for idx,fn in enumerate(list_files):
        file_path = os.path.join(dir,fn)
        # print(fn)
        onesubsession_data = sio.loadmat(file_path)     #keys: data,n_points
        EEG_data = onesubsession_data['data']   #(channels,tot_n_points)  (62,tot_n_points)
        n_points = np.squeeze(onesubsession_data['n_points']).astype(int)
        # print(EEG_data.shape)
        n_points_cum = np.concatenate((np.array([0]),np.cumsum(n_points)))
        n_samples_onesubsession = ((n_points-points_len)//points_step+1).astype(int)
        
        # if idx < n_session:
        #     if idx == s[idx]:
        #         n_samples_onesub = n_samples_onesub + list(n_samples_onesubsession)
        for vid in range(n_vids):
            # print('vid:',vid)
            for i in range(n_samples_onesubsession[vid]):
                # print('sample:',i)

                data[cnt] = EEG_data[:,n_points_cum[vid]+i*points_step:n_points_cum[vid]+i*points_step+points_len]
                cnt+=1

                # 拼接速度会越来越慢
                # temp = temp.reshape(1,temp.shape[0],temp.shape[1])
                # start_time = time.time()
                # data = np.concatenate((data,temp),0)
                # end_time = time.time()
                # print(end_time - start_time)
    # print(cnt)

    n_samples_onesub = np.array(n_samples_onesub)
    n_samples_sessions = n_samples_onesub.reshape(n_session,-1)
    label = [4, 1, 3, 2, 0] * 3 + [2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0] * 2
    onesub_label = []
    for i in range(len(label)):
        onesub_label = onesub_label + [label[i]]*n_samples_onesub[i]
    
    print('load processed data finished!')   

    return data, np.array(onesub_label), n_samples_onesub, n_samples_sessions

def load_processed_SEEDV_NEW_data(dir, fs, n_chans, timeLen, timeStep, n_session=3, n_subs=16, n_vids = 15, n_class=5):
    # input data shape(onesub_onesession):(channels,tot_time) tot_time = sum(eachvids_n_points) 
    # *input data shape（onesub_3session):(channels,tot_time)
    # output : (subs*sum(n_samples_onesub))*channals*time
    #           (16*(sum(n_samples_onesub)))*62*point_len(1250)
    
    
    list_files = os.listdir(dir)
    # list_files.sort(key= lambda x:int(x[:-4]))
    list_files = sorted(list_files, key=lambda x: int(re.search(r'\d+', x).group()))

    # print(list_files)

    points_len = int(timeLen*fs)
    points_step = int(timeStep*fs)


    n_samples_onesub = []
    for fn in list_files:
        file_path = os.path.join(dir,fn)
        onesub_data = sio.loadmat(file_path)  
        n_time = np.squeeze(onesub_data['merged_n_samples_one']).astype(int)
        n_points = np.array(n_time) * fs
        # n_points = np.squeeze(onesubsession_data['n_points']).astype(int)
        n_samples_onesub = ((n_points-points_len)//points_step+1).astype(int)
        # n_samples_onesub = n_samples_onesub + list(n_samples_onesubsession)

    n_samples_sum_onesub = np.sum(n_samples_onesub)


    data = np.empty((n_subs*n_samples_sum_onesub,n_chans,points_len),float)

    s = np.arange(n_session)
    # n_samples_onesub = []
    cnt = 0
    for idx,fn in enumerate(list_files):
        file_path = os.path.join(dir,fn)
        # print(fn)
        onesubsession_data = sio.loadmat(file_path)     #keys: data,n_points
        EEG_data = onesubsession_data['merged_data_all_cleaned']   #(channels,tot_n_points)  (62,tot_n_points)
        thr = 30 * np.median(np.abs(EEG_data))
        EEG_data = (EEG_data - np.mean(EEG_data[EEG_data<thr])) / np.std(EEG_data[EEG_data<thr])
        # n_points = np.squeeze(onesubsession_data['n_points']).astype(int)
        # print(EEG_data.shape)
        n_points_cum = np.concatenate((np.array([0]),np.cumsum(n_points)))
        n_samples_onesub = ((n_points-points_len)//points_step+1).astype(int)

        # if idx < n_session:
        #     if idx == s[idx]:
        #         n_samples_onesub = n_samples_onesub + list(n_samples_onesubsession)
        n_vids = 45
        for vid in range(n_vids):
            # print('vid:',vid)
            for i in range(n_samples_onesub[vid]):
                # print('sample:',i)

                data[cnt] = EEG_data[:,n_points_cum[vid]+i*points_step:n_points_cum[vid]+i*points_step+points_len]
                cnt+=1

                # 拼接速度会越来越慢
                # temp = temp.reshape(1,temp.shape[0],temp.shape[1])
                # start_time = time.time()
                # data = np.concatenate((data,temp),0)
                # end_time = time.time()
                # print(end_time - start_time)
    # print(cnt)

    n_samples_onesub = np.array(n_samples_onesub)
    n_samples_sessions = n_samples_onesub.reshape(n_session,-1)
    label = [4, 1, 3, 2, 0] * 3 + [2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0] * 2
    onesub_label = []
    for i in range(len(label)):
        onesub_label = onesub_label + [label[i]]*n_samples_onesub[i]   

    return data, np.array(onesub_label), n_samples_onesub, n_samples_sessions


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



def test_load_processed_SEEDV_data():
    data_dir = '/mnt/data/model_weights/grm/SEEDV/EEG_processed_sxk'
    # data_dir = 'D:/graduate/G2/xinke/SEEDV/EEG_processed_sxk'
    data_dir2 = '/mnt/data/model_weights/grm/SEEDV/EEG_processed_sampled'
    # data_dir2 = 'D:/graduate/G2/xinke/SEEDV/EEG_processed_sampled'
    timeLen = 5
    timeStep = 2
    fs = 250
    n_channs = 62
    n_session = 3

    data, onesub_label, n_samples_onesub, n_samples_sessions = load_processed_SEEDV_data(data_dir,fs,n_channs,timeLen,timeStep,n_session)
    print(data.shape)
    print(onesub_label)
    print(n_samples_onesub)
    print(n_samples_sessions)
    sampled_data = {}
    sampled_data['data'] = data
    sampled_data['onesub_label'] = onesub_label
    sampled_data['n_samples_onesub'] = n_samples_onesub
    sampled_data['n_samples_sessions'] = n_samples_sessions

def test_load_processed_SEEDV_NEW_data():
    data_dir = '/mnt/data/model_weights/grm/SEEDV-NEW/processed_data_new'
    # data_dir = 'D:/graduate/G2/xinke/SEEDV/EEG_processed_sxk'
    data_dir2 = '/mnt/data/model_weights/grm/SEEDV/EEG_processed_sampled'
    # data_dir2 = 'D:/graduate/G2/xinke/SEEDV/EEG_processed_sampled'
    timeLen = 5
    timeStep = 2
    fs = 125
    n_channs = 60
    n_session = 3

    data, onesub_label, n_samples_onesub, n_samples_sessions = load_processed_SEEDV_NEW_data(data_dir,fs,n_channs,timeLen,timeStep,n_session)
    print(data.shape)
    print(onesub_label)
    print(n_samples_onesub)
    print(n_samples_sessions)
    sampled_data = {}
    sampled_data['data'] = data
    sampled_data['onesub_label'] = onesub_label
    sampled_data['n_samples_onesub'] = n_samples_onesub
    sampled_data['n_samples_sessions'] = n_samples_sessions

# def test_TrainSampler_SEEDV():
#     parser = argparse.ArgumentParser(description=config.DESCRIPTION)
    
#     args = parser.parse_args()
#     args.n_class = config.N_CLASS
#     args.fs = config.FS
#     args.n_channs = config.N_CHANNS 
#     args.n_vids = config.N_VIDS
#     args.t_period =  config.T_PERIOD       
#     args.timeLen = config.TIMELEN           
#     args.timeStep = config.TIMESTEP          
#     args.data_len = args.fs * args.timeLen
#     args.n_subs = config.N_SUBS
#     args.timeLen2 = config.TIMELEN2
#     args.timeStep2 = config.TIMESTEP2
#     args.n_session = config.N_SESSION  #   config设置

#     args.data_dir = config.DATA_DIR          
#     print('data_dir:', args.data_dir)

#     val_vid_inds = np.arange(args.n_vids)
#     train_vid_inds = np.arange(args.n_vids)
#     train_vid_inds = list(train_vid_inds)
#     val_vid_inds = list(val_vid_inds)
#     args.train_vid_inds = train_vid_inds
#     args.val_vid_inds = val_vid_inds
#     args.n_vids_train = len(train_vid_inds)
#     args.n_vids_val = len(val_vid_inds)




#     val_sub = [0]
#     train_sub = list(set(np.arange(args.n_subs)) - set(val_sub))
#     args.train_sub = train_sub
#     args.val_sub = val_sub
#     print('train sub:',args.train_sub)
#     print('val sub:',args.val_sub)
#     data, onesub_label, n_samples_onesub, n_samples_sessions = load_processed_SEEDV_data(
#             args.data_dir, args.fs, args.n_channs, args.timeLen, args.timeStep,args.n_session,args.n_subs,args.n_vids,args.n_class)
#     # data.shape = (15*(sum(n_samples_onesub)))*62*point_len       (n_sub*sum_samples_onesub)*62*1250
#     print('load ext data finished!')
#     # print(data.shape)
    
#     data = data.reshape(args.n_subs, -1, data.shape[-2],data.shape[-1])   # (subs,n_vids*n_samples,n_channs,n_points)




#     # extract对比学习部分（5，2）
#     data_train = data[list(train_sub)].reshape(-1, data.shape[-2], data.shape[-1])   # (train_subs*n_vids*30, 32, 250)
#     data_val = data[list(val_sub)].reshape(-1, data.shape[-2], data.shape[-1])
#     label_train = np.tile(onesub_label, len(train_sub))
#     label_val = np.tile(onesub_label, len(val_sub))
#     trainset = FACED_Dataset(data_train, label_train)
#     valset = FACED_Dataset(data_val, label_val)
#     del data
#     del onesub_label
#     print(data_val.shape)
#     print(label_val.shape)


#     train_sampler = TrainSampler_SEEDV(n_subs=len(train_sub), batch_size=args.n_vids_train,
#                                         n_samples_session=n_samples_sessions, n_session=args.n_session, n_times=1)
#     val_sampler = TrainSampler_SEEDV(n_subs=len(val_sub), batch_size=args.n_vids_val,
#                                     n_samples_session=n_samples_sessions, n_session=args.n_session, n_times=1)

#     # print(train_sampler.__iter__())
#     # print(train_sampler.__len__())

#     train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, pin_memory=True, num_workers=8)
#     val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, pin_memory=True, num_workers=8)


#     for count, (data,labels) in enumerate(train_loader):
#         if count == 0:
#             print(data)
#             print(labels)
#         # print(labels)
#         # print(data.shape)
#     for count, (data,labels) in enumerate(train_loader):
#         if count == 0:
#             print(data)
#             print(labels)

#     print(count)    


# def test_mlp_load():
#     parser = argparse.ArgumentParser(description=config.DESCRIPTION)
    
#     args = parser.parse_args()
#     args.n_class = config.N_CLASS
#     args.fs = config.FS
#     args.n_channs = config.N_CHANNS 
#     args.n_vids = config.N_VIDS
#     args.t_period =  config.T_PERIOD       
#     args.timeLen = config.TIMELEN           
#     args.timeStep = config.TIMESTEP          
#     args.data_len = args.fs * args.timeLen
#     args.n_subs = config.N_SUBS
#     args.timeLen2 = config.TIMELEN2
#     args.timeStep2 = config.TIMESTEP2
#     args.n_session = config.N_SESSION  #   config设置

#     args.data_dir = config.DATA_DIR          
#     print('data_dir:', args.data_dir)

#     val_vid_inds = np.arange(args.n_vids)
#     train_vid_inds = np.arange(args.n_vids)
#     train_vid_inds = list(train_vid_inds)
#     val_vid_inds = list(val_vid_inds)
#     args.train_vid_inds = train_vid_inds
#     args.val_vid_inds = val_vid_inds
#     args.n_vids_train = len(train_vid_inds)
#     args.n_vids_val = len(val_vid_inds)

#     args.normTrain = config.NORMTRAIN




#     val_sub = [0]
#     train_sub = list(set(np.arange(args.n_subs)) - set(val_sub))
#     args.train_sub = train_sub
#     args.val_sub = val_sub
#     print('train sub:',args.train_sub)
#     print('val sub:',args.val_sub)
#     data2, onesub_label2, n_samples2_onesub, n_samples2_sessions = load_processed_SEEDV_data(
#         args.data_dir, args.fs, args.n_channs, args.timeLen2,args.timeStep2,args.n_session,args.n_subs,args.n_vids,args.n_class)
#     print('load mlp data finished!')
#     data2 = data2.reshape(args.n_subs, -1, data2.shape[-2], data2.shape[-1])   # (subs,vid*n_samples, 62, 1250)
#     print(data2.shape)
#     print(onesub_label2.shape)
#     args.n_samples2_onesub = n_samples2_onesub
#     args.n_samples2_sessions = n_samples2_sessions
    
#     # mlp预测部分（1，1）
#     data2_train = data2[list(train_sub)]
#     data2_val = data2[list(val_sub)]
#     del data2
#     # 特征提取前norm Train
#     if args.normTrain:
#         print('normTrain')
#         temp = np.transpose(data2_train,(0,1,3,2))
#         temp = temp.reshape(-1,temp.shape[-1])
#         data2_mean = np.mean(temp, axis=0)
#         data2_var = np.var(temp, axis=0)
#         data2_train = (data2_train - data2_mean.reshape(-1,1)) / np.sqrt(data2_var + 1e-5).reshape(-1,1)
#         data2_val = (data2_val - data2_mean.reshape(-1,1)) / np.sqrt(data2_var + 1e-5).reshape(-1,1)
#         del temp
#     else:
#         print('Do no norm')
#     data2_train = data2_train.reshape(-1, data2_train.shape[-2], data2_train.shape[-1])
#     data2_val = data2_val.reshape(-1, data2_val.shape[-2], data2_val.shape[-1])
#     label2_train = np.tile(onesub_label2, len(train_sub))
#     label2_val = np.tile(onesub_label2, len(val_sub))
#     del onesub_label2
#     train_bs2 = args.n_vids_train *int(np.mean(n_samples2_onesub))
#     trainset2 = FACED_Dataset(data2_train, label2_train)
#     train_loader2 = DataLoader(dataset=trainset2, batch_size=train_bs2, pin_memory=True, num_workers=8,shuffle=False)
#     val_bs2 = max(min(args.n_vids_val *int(np.mean(n_samples2_onesub)),len(label2_val)),1)
#     valset2 = FACED_Dataset(data2_val, label2_val)
#     val_loader2 = DataLoader(dataset=valset2, batch_size=val_bs2, pin_memory=True, num_workers=8,shuffle=False)

#     print(trainset2.__len__())
#     print(train_loader2.__len__())
#     print(data2_val.shape)
#     print(label2_val.shape)


if __name__ == '__main__':
    
    test_load_processed_SEEDV_data()
    test_load_processed_SEEDV_NEW_data()
    
    # pass
    # test_TrainSampler_SEEDV()
    # test_mlp_load()

    
