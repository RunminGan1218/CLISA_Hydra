import torch
import shutil
import numpy as np
import scipy.io as sio


def save_checkpoint(model,optimizer,is_best=False, save_path='checkpoint.pth.tar',copy_path='model_best.pth.tar'):
    print('=> Saving checkpoint')
    checkpoint = {
        'state_dict' : model.state_dict(),
        'optimizer' : optimizer.state_dict(),
    }
    torch.save(checkpoint,save_path)
    if is_best:
        shutil.copyfile(save_path, copy_path)

def copy_checkpoint(save_path,copy_path):
    print('=> Copying checkpoint')
    shutil.copyfile(save_path, copy_path)


def get_confusionMat(output, target, n_class):
    with torch.no_grad():
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        confusionMat = torch.zeros((n_class, n_class))
        for i in range(n_class):
            for j in range(n_class):
                confusionMat[i, j] = torch.sum((pred==i) & (target==j))
        return confusionMat
    
def save_params2mat(load_path,save_path):
    net = torch.load(load_path,map_location='cuda:0') 
    # print(len(net))   
    # print(net.keys())
    state_dict = net['state_dict']
    # print(state_dict.keys())
    timeWeight = state_dict['timeConv.weight'].cpu().numpy()
    spatialWeight1 = state_dict['msConv1.weight'].cpu().numpy()
    spatialWeight2 = state_dict['msConv2.weight'].cpu().numpy()
    spatialWeight3 = state_dict['msConv3.weight'].cpu().numpy()
    spatialWeight4 = state_dict['msConv4.weight'].cpu().numpy()
    # print(timeWeight.shape)  # (kernal_num,group_input_channel(C),H(kernal_size),W(kernal_size))  (16, 1, 1, 60)
    timeWeight = np.squeeze(timeWeight)
    # print(timeWeight.shape)  # (16, 60) (kernal_num,W(1-D,1-chann))


    # print(spatialWeight1.shape)  # (kernal_num,group_input_channel(C),H(kernal_size),W(kernal_size))  (64, 1, 32, 3)
    spatialWeight1 = np.squeeze(spatialWeight1)
    # print(spatialWeight1.shape)  # (64, 32, 3)  (kernal_num(16*4),H,W(2-D,16-chann))

    spatialWeight2 = np.squeeze(spatialWeight2)
    spatialWeight3 = np.squeeze(spatialWeight3)
    spatialWeight4 = np.squeeze(spatialWeight4)
    params = {'timeWeight': timeWeight, 'spatialWeight1': spatialWeight1, 'spatialWeight2': spatialWeight2, 'spatialWeight3': spatialWeight3, 'spatialWeight4': spatialWeight4}
    sio.savemat(save_path, params)

