import torch
import numpy as np

# GPU choose
GPU_INDEX = '7'
GROUP_GPU_INDEX = [2,3]
SERVER = 4
# run number
NO = 3

# dataset choose
DATASET = 'SEEDV'

# model choose
MODEL = 'att'
HAS_ATT = True
GLOBAL_ATT = False

# general train setting

TRAINING_MODE = 'normal'   #cross_training or normal
GENERALIZATIONTEST = False

# Detailed train settings
NORMTRAIN = True
USE_BEST_PRETRAIN = True
EXTRACT_MODE = 'de'

# saving setting
SAVE_LATEST = False

# normal train parameter:
MAX_TOL_NORMAL_EXT = 10
MAX_TOL_NORMAL_MLP = 30
EPOCH_EXT_NORMAL = 30
EPOCH_MLP_NORMAL = 100 

# training mode setting
# VALID_METHOD = '10-folds'
VALID_METHOD = 'loo'
# VALID_METHOD = 'all'

# all_data training setting
if VALID_METHOD == 'all':
    EPOCH_EXT_NORMAL = 4 
    EPOCH_MLP_NORMAL = 12 
    USE_BEST_PRETRAIN = False
    SAVE_LATEST = True



# cross train parameter:
ROUNDS = 8
EPOCH_EXT = 8
EPOCH_MLP = 0
MAX_TOL = 1

PARA_EVOLVE1 = True
PARA_EVOLVE2 = True
LAMBDA1 = 0         # self-superviced CL loss cof
LAMBDA2 = 1         # superviced Clsloss cof
# ALPHA = 1          # uperviced CL loss cof

# model parameters
# extractor
N_TIMEFILTERS = 16
N_MSFILTERS = 4
TIMEFILTERLEN = 60
MSFILTERLEN = 3
MULTIFACT = 2
AVGPOOLLEN = 30
TIMESMOOTHERLEN = 6
SEG_ATT = 30
DILATION_ARRAY=np.array([1,6,12,24])
# DILATION_ARRAY=np.array([1,1,1,1])
STRATIFIED = ['initial', 'middle1', 'middle2']
ACTIV = 'softmax'
EXT_TEMPERATURE = 1.0
SAVEFEA = False

# mlp
HIDDEN_DIM = 128


# training parameters
# EPOCHS_FINETUNE = 100
SAVE_MODEL = False
NUM_WORKERS = 8
BATCH_SIZE = 512
RANDSEED = 7

EXT_LR = 0.0007
MLP_LR = 0.0005
EXT_WEIGHT_DECAY = 0.00015
MLP_WEIGHT_DECAY = 0.0022  #0.001-0.005

NORM_DECAY_RATE = 0.990
LOSS_TEMPERATURE = 0.07


# data config
if DATASET == 'SEEDV':
    N_CLASS = 5
    N_SUBS = 16
    N_SESSION = 3
     
    N_CHANNS = 62
    FS = 250

    TIMELEN = 22      
    TIMESTEP = 11      
    
    TIMELEN2 = 1
    TIMESTEP2 = 1

    T_PERIOD = 0

    N_VIDS = 15

elif DATASET == 'FACED':
    N_CLASS = 9
    N_SUBS = 123
    N_SESSION = 1

    N_CHANNS = 32
    FS = 250

    TIMELEN = 5
    TIMESTEP = 2
    TIMELEN2 = 1
    TIMESTEP2 = 1
    
    T_PERIOD = 30

    # 2/9 classification config
    if N_CLASS==9:
        N_VIDS = 28
        VAL_VID_INDS = np.array([2,5,8,11,14,15,18,21,24,27])
    elif N_CLASS==2:
        N_VIDS = 24
        VAL_VID_INDS = np.arange(2,24,3)




MODEL_CHOOSE = {
    'MODEL': MODEL,
    'HAS_ATT': HAS_ATT,
    'GLOBAL_ATT': GLOBAL_ATT,
}

TRAIN_SETTING = {
    'N_CLASS': N_CLASS,
    'TRAINING_MODE' : TRAINING_MODE,
    'GENERALIZATIONTEST' : GENERALIZATIONTEST,
    'NORMTRAIN' : NORMTRAIN,
    'USE_BEST_PRETRAIN' : USE_BEST_PRETRAIN,
    'EXTRACT_MODE': EXTRACT_MODE,
}

MODEL_SETTING = {
    'MSFILTERLEN' : MSFILTERLEN,
    'DILATION_ARRAY': DILATION_ARRAY.tolist(),
}

EXPERIMENT = {
    'MODEL_CHOOSE' : MODEL_CHOOSE,
    'TRAIN_SETTING' : TRAIN_SETTING,
    'MODEL_SETTING' : MODEL_SETTING,
}







# match SERVER:
#     case 1:
#         DATA_DIR = '/home/ganrunmin/FACED/processed_data'
#         ROOT_DIR = '/home/ganrunmin/data/model_weights/grm/FACED' #song1
#     case 2:
#         DATA_DIR = '/mnt/grm/FACED/processed_data' 
#         ROOT_DIR = '/mnt/data/model_weights/grm/FACED'  #song4
#     case 3:
#         DATA_DIR = '/home/ganrunmin/processed_data'
#         ROOT_DIR = '/mnt/data/ganrunmin/FACED/'
#     case 4:
#         DATA_DIR = '/mnt/data/model_weights/grm/FACED/processed_data'
#         ROOT_DIR = '/mnt/data/model_weights/grm/FACED'  #song4
#     case _:
#         pass    
if DATASET == 'SEEDV':
    DATA_DIR = '/mnt/data/model_weights/grm/SEEDV/EEG_processed_sxk'
    ROOT_DIR = '/mnt/data/model_weights/grm/SEEDV'
elif DATASET == 'FACED':
    DATA_DIR = '/mnt/data/model_weights/grm/FACED/processed_data'
    ROOT_DIR = '/mnt/data/model_weights/grm/FACED'

if TRAINING_MODE == 'ct':
    MODEL_DIR = 'cross_training'
elif TRAINING_MODE == 'normal':
    MODEL_DIR = 'normal_training'
if MODEL == 'att':
    RUN_NO = 'att_'
elif MODEL == 'baseline':
    RUN_NO = 'baseline_'
RUN_NO = RUN_NO + EXTRACT_MODE + '_run'+ str(NO)




DESCRIPTION = 'Parameters of a normal/cross training framework for EEG emotion recognition'





