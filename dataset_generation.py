#%%
import numpy as np
import math
import scipy.io as sio
import matplotlib.pyplot as plt
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
import torch
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
from torch.nn import Module
import DeepMIMO
import argparse
import time
import pickle
from os.path import exists

# torch.cuda.empty_cache()

parser = argparse.ArgumentParser()
parser.add_argument('--measurement_gain', type=float, default=1)
parser.add_argument('--Tx_power_dBm', type=int, default=20)
parser.add_argument('--gpu',type=int,default=0)
parser.add_argument('--scenario',type=str,default='O1_28B')
parser.add_argument('--activated_BS',type=int,default=2)
parser.add_argument('--h_NMSE_dB',type=int,default=-np.inf)
parser.add_argument('--dataset_split_seed',type=int,default=7)
parser.add_argument('--num_feedback',default=None)
parser.add_argument('--UE_rotation', dest='random_UE_rotation', default=False, action='store_true')
parser.add_argument('--use_specific_Tx_power', dest='use_default_Tx_power', default=True, action='store_false')
parser.add_argument('--array_type',type=str,default='ULA')
parser.add_argument('--BW',type=float,default=100,help='MHz')
parser.add_argument('--noise_PSD_dB',type=float,default=-161,help='dBm/Hz')

args = parser.parse_args('')

# if args.use_default_Tx_power:
#     if args.scenario == 'O1_28':
#         tx_power_dBm = 20
#     elif args.scenario == 'O1_28B':
#         tx_power_dBm = 20 
#     elif args.scenario == 'I3_60':
#         if args.activated_BS==1:
#             tx_power_dBm = 15
#         elif args.activated_BS==2:
#             tx_power_dBm = 20
#         else:
#             raise Exception("Unsupported BS activation!") 
#     elif args.scenario == 'Boston5G_28':
#         tx_power_dBm = 40
#     else:
#         raise Exception("Unsupported Ray-Tracing Scenario!")
# else:
#     tx_power_dBm = args.Tx_power_dBm

# noise_power_dBm = args.noise_PSD_dB + 10*np.log10(args.BW*1e6) # dBm
# measurement_gain = args.measurement_gain
# measurement_gain_dB = 10*np.log10(measurement_gain)
# measurement_noise_power = 10**((noise_power_dBm-tx_power_dBm)/10)/measurement_gain


if args.num_feedback is None:
    num_feedback = args.num_feedback
else:
    num_feedback = int(args.num_feedback)
    
gpu_name = "cuda:{}".format(args.gpu)
device = torch.device(gpu_name if torch.cuda.is_available() else 'cpu')
print(device)

if args.scenario == 'O1_28':
    BS_arr = np.array([3])
    ue_row_first = 800
    ue_row_last = 1200
elif args.scenario == 'O1_28B':
    BS_arr = np.array([3])
    ue_row_first = 1
    ue_row_last = 2751    
elif args.scenario == 'I3_60':
    BS_arr = np.array([args.activated_BS])
    # LOS grid: 1-551; NLOS grid: 552-1159
    ue_row_first = 1
    ue_row_last = 551
elif args.scenario == 'Boston5G_28':
    BS_arr = np.array([1])
    ue_row_first = 1
    ue_row_last = 1622
else:
    raise Exception("Unsupported Ray-Tracing Scenario!")
    
parameters = DeepMIMO.default_params()
parameters['dataset_folder'] = '/root/bkup/scenarios'
parameters['scenario'] = args.scenario
parameters['num_paths'] = 25
parameters['active_BS'] = BS_arr
parameters['user_row_first'] = ue_row_first
parameters['user_row_last'] = ue_row_last
parameters['row_subsampling'] = 1
parameters['user_subsampling'] = 0.2

if args.array_type == 'UPA':
    parameters['bs_antenna']['shape'] = np.array([1, 8, 8])
    # parameters['ue_antenna']['shape'] = np.array([1, 6, 6])
elif args.array_type == 'ULA':
    parameters['bs_antenna']['shape'] = np.array([1,64,1])
    # parameters['ue_antenna']['shape'] = np.array([36, 1, 1])
else:
    raise Exception("Unsupported Antenna Array Type!")


parameters['bs_antenna']['spacing'] = 0.5
parameters['bs_antenna']['radiation_pattern'] = 'isotropic'  

parameters['ue_antenna']['shape'] = np.array([1, 1, 1])
# parameters['ue_antenna']['spacing'] = 0.5
parameters['ue_antenna']['radiation_pattern'] = 'isotropic'

# if args.random_UE_rotation and args.array_type == 'UPA':
#     parameters['ue_antenna']['rotation'] = np.array([[-180,180],[-90,90],[-90,90]])
# if args.random_UE_rotation and args.array_type == 'ULA':
#     parameters['ue_antenna']['rotation'] = np.array([[-180,180],[0,0],[0,0]])


parameters['enable_BS2BS'] = 0

parameters['OFDM_channels'] = 0 # Frequency (OFDM) or time domain channels
parameters['OFDM']['subcarriers'] = 8
parameters['OFDM']['subcarriers_limit'] = 1
parameters['OFDM']['subcarriers_sampling'] = 1
parameters['OFDM']['bandwidth'] = args.BW/1e3 # GHz
parameters['OFDM']['RX_filter'] = 0

dataset_savefname = './Dataset/downsampling_5_test_{}_BS{}x{}x{}_UE{}x{}x{}.npy'.format(args.scenario,
                                                parameters['bs_antenna']['shape'][0],
                                                parameters['bs_antenna']['shape'][1],
                                                parameters['bs_antenna']['shape'][2],
                                                parameters['ue_antenna']['shape'][0],
                                                parameters['ue_antenna']['shape'][1],
                                                parameters['ue_antenna']['shape'][2])
if exists(dataset_savefname):
    print('Loading dataset...')
    h = np.load(dataset_savefname,allow_pickle=True)
else:
    print('Generating dataset...')
    dataset = DeepMIMO.generate_data(parameters)
    # h = dataset[0]['user']['channel'].squeeze(axis=-1)
    h = dataset[0]['user']['channel']
    valid_ue_idc = np.array([ue_i for (ue_i,ue_h) in enumerate(h) if not (ue_h==0).all()])
    print('Keep {} out of {} UEs that have valid paths.'.format(len(valid_ue_idc),h.shape[0]))
    h = h[valid_ue_idc]
    h = np.sum(h,axis=3).squeeze()
    np.save(dataset_savefname,h,allow_pickle=True)
