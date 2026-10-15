# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 19:26:34 2021

@author: Yuqiang (Ethan) Heng
"""
#%%
import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from beam_utils_L import ULA_DFT_codebook as DFT_codebook

np.random.seed(7)
# number of narrow data beams to select from (N_V)
n_nb = 64
# number of probing beams (N_W)
n_wide_beams = [4,8,12,16,20,24,28,32]
n_nb=64
n_antenna = 64
noise_power_dBm = -81
noiseless = False
noise_factor = 0 #dB

dataset_name = 'Rosslyn_ULA' # 'Rosslyn_ULA' or 'O1_28B_ULA' or 'I3_60_ULA' or 'O1_28_ULA'
# Training and testing data:
if dataset_name == 'Rosslyn_ULA':
    h_real = np.load('./Dataset/Rosslyn/MISO_Static_FineGrid_Hmatrices_real.npy')
    h_imag = np.load('./Dataset/Rosslyn/MISO_Static_FineGrid_Hmatrices_imag.npy')
    tx_power_dBm = 10
    h = h_real + 1j*h_imag
    valid_ue_idc = np.array([row_idx for (row_idx,row) in enumerate(np.concatenate((h_real,h_imag),axis=1)) if not all(row==0)])
    h = h[valid_ue_idc]
    h_real = h_real[valid_ue_idc]
    h_imag = h_imag[valid_ue_idc]
elif dataset_name == 'O1_28B_ULA':
    h = np.load('./Dataset/test_O1_28B_BS1x64x1_UE1x1x1.npy')
    h=h.squeeze()
    h_real = np.real(h)
    h_imag = np.imag(h)
    tx_power_dBm = 20
elif dataset_name == 'I3_60_ULA':
    fname_h_real = './Dataset/DeepMIMO Dataset/I3_60_ULA/h_real.mat'
    fname_h_imag = './Dataset/DeepMIMO Dataset/I3_60_ULA/h_imag.mat'
    h_real = sio.loadmat(fname_h_real)['h_real']
    h_imag = sio.loadmat(fname_h_imag)['h_imag']
    tx_power_dBm = 10
elif dataset_name == 'O1_28_ULA':
    h = np.load('./Dataset/test_O1_28_BS1x64x1_UE1x1x1.npy')
    h=h.squeeze()
    h_real = np.real(h)
    h_imag = np.imag(h)
    tx_power_dBm = 10
else:
    raise NameError('Dataset Not Supported')   

if noiseless:
    noise_power_dBm = -np.inf    
noise_power = 10**((noise_power_dBm-tx_power_dBm-noise_factor)/10)


norm_factor = np.max(abs(h))

h_scaled = h/norm_factor
h_concat_scaled = np.concatenate((h_real/norm_factor,h_imag/norm_factor),axis=1)

train_idc, test_idc = train_test_split(np.arange(h.shape[0]),test_size=0.4)
val_idc, test_idc = train_test_split(test_idc,test_size=0.5)
dft_nb_codebook = DFT_codebook(nseg=n_nb,n_antenna=n_antenna)
label = np.argmax(np.power(np.absolute(np.matmul(h_scaled, dft_nb_codebook.conj().T)),2),axis=1)

x_train,y_train = h_scaled[train_idc,:],label[train_idc]
x_val,y_val = h_scaled[val_idc,:],label[val_idc]
x_test,y_test = h_scaled[test_idc,:],label[test_idc]

print(dataset_name)
for codebook_type in ['Ethan','complex_proposed','magnitude_proposed','DFT']:
    if codebook_type == 'trainable':
        print_prefix = '& learned'
    else:
        print_prefix = '& '+codebook_type
    s_scores = []
    for i,n_wb in enumerate(n_wide_beams):    
        print(n_wb)        
        if codebook_type == 'Ethan':
            probing_codebook = np.load('./Saved Codebooks/Ethan_{}_{}_beam_probing_codebook_{}_beam_noise_{}_dBm_tx_power_{}_dBm.npy'.format(dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm)).T
    
        elif codebook_type == 'complex_proposed':
            probing_codebook = np.load('./Saved Codebooks/complex_feedback_proposed_codebook_{}_{}_beam_probing_codebook_{}_beam_noise_{}_dBm_tx_power_{}_dBm.npy'.format(dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm))
        elif codebook_type == 'magnitude_proposed':
            probing_codebook = np.load('./Saved Codebooks/magnitude_feedback_proposed_codebook_{}_{}_beam_probing_codebook_{}_beam_noise_{}_dBm_tx_power_{}_dBm.npy'.format(dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm))
        elif codebook_type == 'DFT':
            probing_codebook = DFT_codebook(nseg=n_wb,n_antenna=n_antenna)
        
        h_project_probing = np.absolute(np.matmul(x_test, probing_codebook.conj().T))
        probing_silhouette_score = silhouette_score(h_project_probing, y_test)
        s_scores.append(probing_silhouette_score)
    s_scores_char = print_prefix
    for i in s_scores:
        s_scores_char += ' & {:.3f}'.format(i)
    if codebook_type == 'DFT':
        s_scores_char += ' \\\\ \\hline'
    else:
        s_scores_char += ' \\\\ \\cline{2-10}'
    print(s_scores_char)
