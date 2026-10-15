#%%
import os
# os.chdir('/root/bkup/Learning_site_specific')
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from ComplexLayers_Torch import Beam_Classifier, projection_trainer,projection_trainer_fake, fit,fit_projector,  eval_model, custom_projection_L
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
from beam_utils_L import ULA_DFT_codebook as DFT_codebook
from beam_utils_L import ULA_DFT_codebook_blockmatrix as DFT_codebook_blockmatrix
from beam_utils_L import plot_codebook_pattern, DFT_angles, codebook_blockmatrix, get_AMCF_codebook, calc_beam_pattern, generate_phase_noise_matrix, generate_noise_matrix
from scipy.io import savemat

np.random.seed(8)
# number of probing beams (N_W)
n_wb=64
n_nb=64

n_antenna = 64
top_k=1
nepoch = 100
batch_size = 500            
noise_factor = 0 #dB
noise_power_dBm = -81
noiseless = False

nb_topk_list=[128] #decreasing order
dataset_name = 'O1_28B_ULA' # 'Rosslyn_ULA' or 'O1_28B_ULA' or 'I3_60_ULA' or 'O1_28_ULA'
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


dft_nb_codebook = DFT_codebook(nseg=n_nb,n_antenna=n_antenna)
topkdiff=[]
for nb_topk_ind, nb_topk in enumerate(nb_topk_list):
    nb_label_clean = np.argmax(np.power(np.absolute(np.matmul(h_scaled, dft_nb_codebook.conj().T)),2),axis=1)
    soft_label = np.power(np.absolute(np.matmul(h, dft_nb_codebook.conj().T)),2)

    nb_measurement_complex = np.matmul(h_scaled, dft_nb_codebook.conj().T)
    noise_nb_measurement_vec = np.random.normal(0,1,(nb_measurement_complex.shape[0],nb_measurement_complex.shape[1]))*np.sqrt(noise_power/2)/norm_factor + 1j*np.random.normal(0,1,(nb_measurement_complex.shape[0],nb_measurement_complex.shape[1]))*np.sqrt(noise_power/2)/norm_factor
    nb_measurement_complex = nb_measurement_complex + noise_nb_measurement_vec
    nb_measurement_abs = np.abs(nb_measurement_complex)


    nb_label = np.argmax(np.power(np.absolute(nb_measurement_complex),2),axis=1)
    nb_measurement_complex_concat = np.concatenate((np.real(nb_measurement_complex),np.imag(nb_measurement_complex)),axis=1)
            
    nb_measurement_topk_ind = np.argsort(nb_measurement_abs,axis=1)[:,-nb_topk:]
    nb_measurement_topk_abs = np.zeros_like(nb_measurement_abs)
    for row_index, topk_indices in enumerate(nb_measurement_topk_ind):
        nb_measurement_topk_abs[row_index,topk_indices] = nb_measurement_abs[row_index,topk_indices]

    nb_measurement_topk_real = np.zeros_like(nb_measurement_abs)
    nb_measurement_topk_imag = np.zeros_like(nb_measurement_abs)

    for row_index, topk_indices in enumerate(nb_measurement_topk_ind):
        nb_measurement_topk_real[row_index,topk_indices] = np.real(nb_measurement_complex)[row_index,topk_indices]
        nb_measurement_topk_imag[row_index,topk_indices] = np.imag(nb_measurement_complex)[row_index,topk_indices]


    nb_measurement_complex_topk_concat = np.concatenate((nb_measurement_topk_real,nb_measurement_topk_imag),axis=1)


    # based on h_scaled 
    soft_label_dB = 10*np.log10(soft_label)
    soft_label_dB_max = np.max(soft_label_dB,axis=1)
    # nb_measurement_abs_dB = 10*np.log10(np.power(nb_measurement_abs,2))
    # nb_measurement_abs_dB_max = np.max(nb_measurement_abs_dB,axis=1)
    # nb_measurement_abs_dB_min = np.min(nb_measurement_abs_dB,axis=1)
    # nb_measurement_abs_dB_diffmax = nb_measurement_abs_dB_max - nb_measurement_abs_dB_min

    nb_measurement_topk_abs_dB = 10*np.log10(np.power(nb_measurement_topk_abs,2))
    nb_measurement_topk_abs_dB_max = np.max(nb_measurement_topk_abs_dB,axis=1)
    nb_measurement_topk_abs_dB_min = np.min(np.where(nb_measurement_topk_abs_dB==-np.inf,np.inf,nb_measurement_topk_abs_dB),axis=1)
    nb_measurement_topk_abs_dB_diffmax = nb_measurement_topk_abs_dB_max - nb_measurement_topk_abs_dB_min
    topkdiff.append(nb_measurement_topk_abs_dB_diffmax)
#%%
for nb_topk_ind, nb_topk in enumerate(nb_topk_list):
    if nb_topk_ind==0:
        plt.plot(topkdiff[nb_topk_ind],color='brown',label='Among all beams')
    if nb_topk_ind==1:
        plt.plot(topkdiff[nb_topk_ind],color='pink',label='Among top-{} strongest beams'.format(nb_topk))
    if nb_topk_ind==2:
        plt.plot(topkdiff[nb_topk_ind],color='gray',label='Among top-{} strongest beams'.format(nb_topk))
    if nb_topk_ind==3:
        plt.plot(topkdiff[nb_topk_ind],color='olive',label='Among top-{} strongest beams'.format(nb_topk))
    if nb_topk_ind==4:
        plt.plot(topkdiff[nb_topk_ind],color='steelblue',label='Among top-{} strongest beams'.format(nb_topk))
        
plt.legend()
            
plt.xlabel('UE index', fontsize=14)  
plt.ylabel('max RSRP - min RSRP (dB)', fontsize=14)  

plt.xticks(fontsize=12) 
plt.yticks(fontsize=12) 
# savemat('./Saved matfile/topkdiff_{}.mat'.format(dataset_name), {'topkdiff':topkdiff})

        # nb_measurement_topk_abs_dB = 10*np.log10(np.power(nb_measurement_topk_abs,2))
        # quantized_nb_measurement_topk_abs_dB = np.round(nb_measurement_topk_abs_dB/dB_stepsize)*dB_stepsize
        # quantized_nb_measurement_topk_abs = np.sqrt(10**(quantized_nb_measurement_topk_abs_dB/10))
#%% Genie-aided average SNR

dft_nb_codebook = DFT_codebook(nseg=n_nb,n_antenna=n_antenna)
nb_bf_signal = np.matmul(h, dft_nb_codebook.conj().T)
nb_bf_noise_real = np.random.normal(loc=0,scale=1,size=nb_bf_signal.shape)*np.sqrt(noise_power/2)
nb_bf_noise_imag = np.random.normal(loc=0,scale=1,size=nb_bf_signal.shape)*np.sqrt(noise_power/2)
nb_bf_signal_with_noise = nb_bf_signal + nb_bf_noise_real + 1j*nb_bf_noise_imag
nb_bf_gain = np.power(np.absolute(nb_bf_signal),2)
nb_bf_gain_with_noise = np.power(np.absolute(nb_bf_signal_with_noise),2)
best_nb_noisy = np.argmax(nb_bf_gain_with_noise,axis=1)
best_nb = np.argmax(nb_bf_gain,axis=1)
exhaustive_acc = (best_nb_noisy==best_nb).mean()
genie_nb_snr = tx_power_dBm + 10*np.log10(nb_bf_gain.max(axis=1)) - noise_power_dBm + noise_factor
exhaustive_nb_snr = np.array([nb_bf_gain[ue_idx,best_nb_idx_noisy] for ue_idx,best_nb_idx_noisy in enumerate(best_nb_noisy)])
exhaustive_nb_snr = tx_power_dBm + 10*np.log10(exhaustive_nb_snr) - noise_power_dBm + noise_factor
print('Genie avg. SNR = {} dB.'.format(genie_nb_snr.mean()))
print('Exhaustive search accuracy = {}, avg. SNR = {} dB.'.format(exhaustive_acc,exhaustive_nb_snr.mean()))


#%%

sorted_snr_dB = soft_label_dB
sorted_snr_dB.sort(axis=1)
sorted_snr_dB = sorted_snr_dB + tx_power_dBm  - noise_power_dBm + noise_factor
# savemat('./Saved matfile/sortedsnr_{}_{}.mat'.format(dataset_name,n_nb), {'sorted_snr_dB':sorted_snr_dB})
#%%