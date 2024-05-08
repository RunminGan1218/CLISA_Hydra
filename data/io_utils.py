import os
import numpy as np
import scipy.io as sio
import re
import pickle

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
    
    print('load processed data finished!')   

    return data, np.array(onesub_label), n_samples_onesub, n_samples_sessions

def load_processed_SEEDV_NEW_data(dir, fs, n_chans, timeLen, timeStep, n_session=3, n_subs=16, n_vids = 15, n_class=5):
    # input data shape(onesub_onesession):(channels,tot_time) tot_time = sum(eachvids_n_points) 
    # *input data shape（onesub_3session):(channels,tot_time)
    # output : (subs*sum(n_samples_onesub))*channals*time
    #           (16*(sum(n_samples_onesub)))*62*point_len(1250)
    

    list_files = os.listdir(dir)
    list_files = sorted(list_files, key=lambda x: int(re.search(r'\d+', x).group()))
    points_len = int(timeLen*fs)
    points_step = int(timeStep*fs)
    
    # 3 session in all change delete the loop
    file_path = os.path.join(dir,list_files[0])
    onesub_data = sio.loadmat(file_path)  
    n_time = np.squeeze(onesub_data['merged_n_samples_one']).astype(int)
    n_points = np.array(n_time) * fs
    n_samples_onesub = ((n_points-points_len)//points_step+1).astype(int)
    n_samples_sum_onesub = np.sum(n_samples_onesub)
    
    data = np.empty((n_subs*n_samples_sum_onesub,n_chans,points_len),float)

    cnt = 0
    for idx,fn in enumerate(list_files):
        file_path = os.path.join(dir,fn)
        # print(fn)
        onesub_data = sio.loadmat(file_path)     #keys: data,n_points
        EEG_data = onesub_data['merged_data_all_cleaned']   #(channels,tot_n_points_3session)  (60,tot_n_points_3session)
        thr = 30 * np.median(np.abs(EEG_data))
        EEG_data = (EEG_data - np.mean(EEG_data[EEG_data<thr])) / np.std(EEG_data[EEG_data<thr])
        n_points_cum = np.concatenate((np.array([0]),np.cumsum(n_points)))

        
        n_vids = 45
        for vid in range(n_vids):
            # print('vid:',vid)
            for i in range(n_samples_onesub[vid]):
                # print('sample:',i)
                data[cnt] = EEG_data[:,n_points_cum[vid]+i*points_step:n_points_cum[vid]+i*points_step+points_len]
                cnt+=1
    
    n_samples_onesub = np.array(n_samples_onesub)
    n_samples_sessions = n_samples_onesub.reshape(n_session,-1)
    label = [4, 1, 3, 2, 0] * 3 + [2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0] * 2
    onesub_label = []
    for i in range(len(label)):
        onesub_label = onesub_label + [label[i]]*n_samples_onesub[i]   
    return data, np.array(onesub_label), n_samples_onesub, n_samples_sessions


def save_sliced_data(sliced_data_dir, data, onesub_labels, n_samples_onesub, n_samples_sessions):
    if not os.path.exists(sliced_data_dir+'/metadata'):
        os.makedirs(sliced_data_dir+'/metadata')
    if not os.path.exists(sliced_data_dir+'/data'):
        os.makedirs(sliced_data_dir+'/data')
    np.save(sliced_data_dir+'/metadata/onesub_labels.npy', onesub_labels)
    np.save(sliced_data_dir+'/metadata/n_samples_onesub.npy', n_samples_onesub)
    np.save(sliced_data_dir+'/metadata/n_samples_sessions.npy', n_samples_sessions)
    for sample in range(data.shape[0]):
        np.save(sliced_data_dir+f'/data/data_sample_{sample}.npy', data[sample])
    np.save(sliced_data_dir+'/saved.npy', [True])
    print('save sliced data finished!')

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

if __name__ == '__main__':
    test_load_processed_SEEDV_data()
    test_load_processed_SEEDV_NEW_data()