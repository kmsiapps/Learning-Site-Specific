# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 20:04:13 2021

@author: Yuqiang (Ethan) Heng
"""
#%%
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from ComplexLayers_Torch import Beam_Classifier, eval_model
from ComplexLayers_Torch import Beam_Classifier, Beam_Classifier_Prune, projection_trainer,projection_trainer_fake, fit,fit_projector,eval_model
import torch.utils.data
import torch.nn as nn
from sklearn.model_selection import train_test_split
from beam_utils_L import ULA_DFT_codebook as DFT_codebook
from beam_utils_L import plot_codebook_pattern, DFT_angles, Beam_Search_Tree, get_AMCF_codebook, AMCF_boundaries
from beam_utils_L import codebook_blockmatrix
from scipy.io import savemat

from tqdm.auto import tqdm

if_fake = True
M = 10 # 0

np.random.seed(7)
# number of probing beams (N_W)
n_wide_beams = [12]
# number of narrow data beams to select from (N_V)
n_narrow_beams = [64 for i in n_wide_beams]
n_antenna = 64
batch_size = 500

noise_power_dBm_list =  [-90,-87,-84,-81,-78,-75,-72,-69] 
noise_factor = 0 #dB

noiseless = False

pruning_beam_idx = {
    'Rosslyn_ULA': [7, 6, 8, 9, 10, 12, 11, 13, 3, 5, 4, 2, 14, 1, 15, 16, 17, 18, 19, 0],
    'O1_28B_ULA': [9, 13, 5, 12, 7, 11, 3, 6, 14, 4, 8, 15, 10, 16, 2, 17, 1, 18, 0, 19]
}
 
dataset_name = os.environ.get('DATASET_NAME', ['Rosslyn_ULA', 'O1_28B_ULA'][1]) # 'Rosslyn_ULA' or 'O1_28B_ULA' or 'I3_60_ULA' or 'O1_28_ULA'
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
    h = np.load('./Dataset/downsampling_5_test_O1_28B_BS1x64x1_UE1x1x1.npy')
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
    fname_h_real = './Dataset/DeepMIMO Dataset/O1_28_ULA/h_real.mat'
    fname_h_imag = './Dataset/DeepMIMO Dataset/O1_28_ULA/h_imag.mat'
    h_real = sio.loadmat(fname_h_real)['h_real']
    h_imag = sio.loadmat(fname_h_imag)['h_imag']
    tx_power_dBm = 10
else:
    raise NameError('Dataset Not Supported')   

if noiseless:
    noise_power_dBm = -np.inf    
top_k=1

norm_factor = np.max(abs(h))
h_scaled = h/norm_factor
h_concat_scaled = np.concatenate((h_real/norm_factor,h_imag/norm_factor),axis=1)

train_idc, test_idc = train_test_split(np.arange(h.shape[0]),test_size=0.4)
val_idc, test_idc = train_test_split(test_idc,test_size=0.5)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Plot evaluation device: {device}')

def predict_numpy(model, x_tensor, batch_size, device):
    model = model.to(device)
    if hasattr(model, 'codebook') and isinstance(model.codebook, torch.Tensor):
        model.codebook = model.codebook.to(device)
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, x_tensor.shape[0], batch_size):
            xb = x_tensor[start:start+batch_size].float().to(device)
            outputs.append(model(xb).detach().cpu())
    model.cpu()
    return torch.cat(outputs, dim=0).numpy()


def topk_metrics_from_sorted(pred_sorted, soft_label_test, noise_power, M):
    pred_top3 = pred_sorted[:, :3]
    gains_top3 = soft_label_test[np.arange(soft_label_test.shape[0])[:, None], pred_top3]
    gt_beams = np.argmax(soft_label_test, axis=1)
    sigma = np.sqrt(noise_power / 2)
    topk_acc = []
    topk_gains = []
    for k in range(1, 4):
        beams_k = pred_top3[:, :k]
        gains_k = gains_top3[:, :k]
        nb = (np.random.normal(0, sigma, size=(M, gains_k.shape[0], k))
              + 1j*np.random.normal(0, sigma, size=(M, gains_k.shape[0], k)))
        noisy = np.abs(np.sqrt(gains_k)[None, :, :] + nb)**2
        chosen_pos = np.argmax(noisy, axis=2)
        chosen_beams = np.take_along_axis(beams_k[None, :, :], chosen_pos[:, :, None], axis=2).squeeze(2)
        chosen_gains = np.take_along_axis(gains_k[None, :, :], chosen_pos[:, :, None], axis=2).squeeze(2)
        topk_acc.append((chosen_beams == gt_beams[None, :]).mean(axis=0).mean())
        topk_gains.append(chosen_gains.mean(axis=0))
    return np.stack(topk_gains, axis=1), np.array(topk_acc)

#%%

dft_codebook_acc = []
dft_prune_codebook_acc = []
learnable_codebook_acc = []
complex_feedback_proposed_codebook_acc = []
magnitude_feedback_proposed_codebook_acc = []
norandomphase_proposed_codebook_acc = []

dft_codebook_topk_acc = []
dft_prune_codebook_topk_acc = []
learnable_codebook_topk_acc = []
complex_feedback_proposed_codebook_topk_acc = []
magnitude_feedback_proposed_codebook_topk_acc=[]
norandomphase_proposed_codebook_topk_acc=[]

dft_codebook_topk_gain = []
dft_prune_codebook_topk_gain = []
learned_codebook_topk_gain= []
complex_feedback_proposed_codebook_topk_gain= []
magnitude_feedback_proposed_codebook_topk_gain = []
norandomphase_proposed_codebook_topk_gain = []

for noise_power_dBm_ind, noise_power_dBm in enumerate(tqdm(noise_power_dBm_list, desc=f"{dataset_name} noise sweep")):
    noise_power = 10**((noise_power_dBm-tx_power_dBm-noise_factor)/10)
    for n_wb_i, n_wb in enumerate(n_wide_beams):
        n_nb = n_narrow_beams[n_wb_i]
        print('{} Wide Beams, {} Narrow Beams.'.format(n_wb,n_nb))
    


    optimal_gains = []

      

    learnable_codebook =np.load('./Saved Codebooks/Ethan_{}_{}_beam_probing_codebook_{}_beam_noise_{}_dBm_tx_power_{}_dBm.npy'.format(dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm))
    complex_feedback_proposed_codebook = np.load('./Saved Codebooks/complex_feedback_proposed_codebook_{}_{}_beam_probing_codebook_{}_beam_noise_{}_dBm_tx_power_{}_dBm.npy'.format(dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm))
    magnitude_feedback_proposed_codebook = np.load('./Saved Codebooks/magnitude_feedback_proposed_codebook_{}_{}_beam_probing_codebook_{}_beam_noise_{}_dBm_tx_power_{}_dBm.npy'.format(dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm))
        
    dft_nb_codebook = DFT_codebook(nseg=n_nb,n_antenna=n_antenna)
    label = np.argmax(np.power(np.absolute(np.matmul(h_scaled, dft_nb_codebook.conj().T)),2),axis=1)
    soft_label = np.power(np.absolute(np.matmul(h, dft_nb_codebook.conj().T)),2)

    x_train,y_train = h_concat_scaled[train_idc,:],label[train_idc]
    x_val,y_val = h_concat_scaled[val_idc,:],label[val_idc]
    x_test,y_test = h_concat_scaled[test_idc,:],label[test_idc]
    
    torch_x_train,torch_y_train = torch.from_numpy(x_train),torch.from_numpy(y_train)
    torch_x_val,torch_y_val = torch.from_numpy(x_val),torch.from_numpy(y_val)
    torch_x_test,torch_y_test = torch.from_numpy(x_test),torch.from_numpy(y_test)
    
    # Pytorch train and test sets
    train = torch.utils.data.TensorDataset(torch_x_train,torch_y_train)
    val = torch.utils.data.TensorDataset(torch_x_val,torch_y_val)
    test = torch.utils.data.TensorDataset(torch_x_test,torch_y_test)
    
    # data loader
    train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)
    val_loader = torch.utils.data.DataLoader(val, batch_size = batch_size, shuffle = False)
    test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)
    
    learnable_codebook_model = Beam_Classifier(n_antenna=n_antenna,n_wide_beam=n_wb,n_narrow_beam=n_nb,
                                            trainable_codebook=True,noise_power=noise_power,norm_factor=norm_factor)
    learnable_model_savefname = './Saved Models/Ethan_{}_{}_beam_probing_codebook_{}_beam_noise_{}_dBm_tx_power_{}_dBm.pt'.format(dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm)
    learnable_codebook_model.load_state_dict(torch.load(learnable_model_savefname))
    
    learnable_codebook_model.codebook.reset_parameters(theta=torch.Tensor(np.angle(learnable_codebook)))

    learnable_codebook_test_loss,learnable_codebook_test_acc = np.nan, np.nan 
    learnable_codebook_acc.append(learnable_codebook_test_acc)
    y_test_predict_learnable_codebook = predict_numpy(learnable_codebook_model, torch_x_test, batch_size, device)
    topk_sorted_test_learned_codebook = (-y_test_predict_learnable_codebook).argsort()
    topk_bf_gain_learnable_codebook, topk_acc_learnable_codebook = topk_metrics_from_sorted(topk_sorted_test_learned_codebook, soft_label[test_idc,:], noise_power, M)
    learned_codebook_topk_gain.append(topk_bf_gain_learnable_codebook)
    learnable_codebook_topk_acc.append(topk_acc_learnable_codebook)

    
    dft_wb_codebook = DFT_codebook(nseg=n_wb,n_antenna=n_antenna)
    dft_codebook_model = Beam_Classifier(n_antenna=n_antenna,n_wide_beam=n_wb,n_narrow_beam=n_nb,
                                        trainable_codebook=False,complex_codebook=dft_wb_codebook,
                                        noise_power=noise_power,norm_factor=norm_factor)
    dft_model_savefname = './Saved Models/DFTprob_{}_{}_beam_probing_codebook_{}_beam_noise_{}_dBm_tx_power_{}_dBm.pt'.format(dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm)
    dft_codebook_model.load_state_dict(torch.load(dft_model_savefname))
    dft_codebook_test_loss,dft_codebook_test_acc = np.nan, np.nan 
    dft_codebook_acc.append(dft_codebook_test_acc)
    y_test_predict_dft_codebook = predict_numpy(dft_codebook_model, torch_x_test, batch_size, device)
    topk_sorted_test_dft_codebook = (-y_test_predict_dft_codebook).argsort()
    topk_bf_gain_dft_codebook, topk_acc_dft_codebook = topk_metrics_from_sorted(topk_sorted_test_dft_codebook, soft_label[test_idc,:], noise_power, M)
    dft_codebook_topk_gain.append(topk_bf_gain_dft_codebook)
    dft_codebook_topk_acc.append(topk_acc_dft_codebook)

    # pruned wide dft model        
    n_wb_before_prune = 20
    dft_wb_codebook = DFT_codebook(nseg=n_wb_before_prune,n_antenna=n_antenna)
    dft_prune_codebook_model = Beam_Classifier_Prune(n_antenna=n_antenna,n_wide_beam=n_wb_before_prune,n_narrow_beam=n_nb,
                                        trainable_codebook=False,complex_codebook=dft_wb_codebook,
                                        noise_power=noise_power,norm_factor=norm_factor, prune_idx=pruning_beam_idx[dataset_name][:n_wb_before_prune-n_wb])
    dft_model_savefname = './Saved Models/DFTprob_{}_{}_beam_probing_codebook_{}_beam_noise_{}.0_dBm_tx_power_{}_dBm.pt'.format(dataset_name,n_wb_before_prune,n_nb,noise_power_dBm,tx_power_dBm)
    dft_prune_codebook_model.load_state_dict(torch.load(dft_model_savefname))
    dft_prune_codebook_test_loss,dft_prune_codebook_test_acc = np.nan, np.nan 
    dft_prune_codebook_acc.append(dft_prune_codebook_test_acc)
    y_test_predict_dft_prune_codebook = predict_numpy(dft_prune_codebook_model, torch_x_test, batch_size, device)
    topk_sorted_test_dft_prune_codebook = (-y_test_predict_dft_prune_codebook).argsort()
    topk_bf_gain_dft_prune_codebook, topk_acc_dft_prune_codebook = topk_metrics_from_sorted(topk_sorted_test_dft_prune_codebook, soft_label[test_idc,:], noise_power, M)
    dft_prune_codebook_topk_gain.append(topk_bf_gain_dft_prune_codebook)
    dft_prune_codebook_topk_acc.append(topk_acc_dft_prune_codebook)

    # =======

    complex_feedback_proposed_codebook_model = Beam_Classifier(n_antenna=n_antenna,n_wide_beam=n_wb,n_narrow_beam=n_nb,
                                        trainable_codebook=False,complex_codebook=complex_feedback_proposed_codebook,
                                        noise_power=noise_power,norm_factor=norm_factor)
    complex_feedback_proposed_model_savefname = './Saved Models/complex_feedback_propsed_{}_{}_beam_probing_codebook_{}_beam_noise_{}_dBm_tx_power_{}_dBm.pt'.format(dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm)
    complex_feedback_proposed_codebook_model.load_state_dict(torch.load(complex_feedback_proposed_model_savefname))

    complex_feedback_projection_training_model = projection_trainer(n_wide_beam=n_wb,n_narrow_beam=n_nb,narrow_beam_CB=dft_nb_codebook,
                                                noise_power=noise_power,norm_factor=norm_factor)
    complex_feedback_projection_training_model_savefname = './Saved Projection Models/complex_feedback_projection_trainer_{}_{}_beam_probing_codebook_{}_beam_noise_{}_dBm_tx_power_{}_dBm.pt'.format(dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm)
    complex_feedback_projection_training_model.load_state_dict(torch.load(complex_feedback_projection_training_model_savefname))
    
    complex_feedback_proposed_codebook_model.dense1.load_state_dict(complex_feedback_projection_training_model.dense1.state_dict())
    complex_feedback_proposed_codebook_model.dense2.load_state_dict(complex_feedback_projection_training_model.dense2.state_dict())
    complex_feedback_proposed_codebook_model.dense3.load_state_dict(complex_feedback_projection_training_model.dense3.state_dict())

    complex_feedback_proposed_codebook_test_loss,complex_feedback_proposed_codebook_test_acc = np.nan, np.nan 
    complex_feedback_proposed_codebook_acc.append(complex_feedback_proposed_codebook_test_acc)
    y_test_predict_complex_feedback_proposed_codebook = predict_numpy(complex_feedback_proposed_codebook_model, torch_x_test, batch_size, device)
    topk_sorted_test_complex_feedback_proposed_codebook = (-y_test_predict_complex_feedback_proposed_codebook).argsort()
    topk_bf_gain_complex_feedback_proposed_codebook, topk_acc_complex_feedback_proposed_codebook = topk_metrics_from_sorted(topk_sorted_test_complex_feedback_proposed_codebook, soft_label[test_idc,:], noise_power, M)
    complex_feedback_proposed_codebook_topk_gain.append(topk_bf_gain_complex_feedback_proposed_codebook)
    complex_feedback_proposed_codebook_topk_acc.append(topk_acc_complex_feedback_proposed_codebook)
    
    magnitude_feedback_proposed_codebook_model = Beam_Classifier(n_antenna=n_antenna,n_wide_beam=n_wb,n_narrow_beam=n_nb,
                                        trainable_codebook=False,complex_codebook=magnitude_feedback_proposed_codebook,
                                        noise_power=noise_power,norm_factor=norm_factor)
    magnitude_feedback_proposed_codebook_model_savefname = './Saved Models/magnitude_feedback_propsed_{}_{}_beam_probing_codebook_{}_beam_noise_{}_dBm_tx_power_{}_dBm.pt'.format(dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm)
    magnitude_feedback_proposed_codebook_model.load_state_dict(torch.load(magnitude_feedback_proposed_codebook_model_savefname))
    
    
    magnitude_feedback_projection_training_model = projection_trainer_fake(if_fake,n_wide_beam=n_wb,n_narrow_beam=n_nb,narrow_beam_CB=dft_nb_codebook,
                                                noise_power=noise_power,norm_factor=norm_factor)
    magnitude_feedback_projection_training_model_savefname = './Saved Projection Models/magnitude_feedback_projection_trainer_{}_{}_beam_probing_codebook_{}_beam_noise_{}_dBm_tx_power_{}_dBm.pt'.format(dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm)
    magnitude_feedback_projection_training_model.load_state_dict(torch.load(magnitude_feedback_projection_training_model_savefname))
    
    magnitude_feedback_proposed_codebook_model.dense1.load_state_dict(magnitude_feedback_projection_training_model.dense1.state_dict())
    magnitude_feedback_proposed_codebook_model.dense2.load_state_dict(magnitude_feedback_projection_training_model.dense2.state_dict())
    magnitude_feedback_proposed_codebook_model.dense3.load_state_dict(magnitude_feedback_projection_training_model.dense3.state_dict())

    
    magnitude_feedback_proposed_codebook_test_loss,magnitude_feedback_proposed_codebook_test_acc = np.nan, np.nan 
    magnitude_feedback_proposed_codebook_acc.append(magnitude_feedback_proposed_codebook_test_acc)
    y_test_predict_magnitude_feedback_proposed_codebook = predict_numpy(magnitude_feedback_proposed_codebook_model, torch_x_test, batch_size, device)
    topk_sorted_test_magnitude_feedback_proposed_codebook = (-y_test_predict_magnitude_feedback_proposed_codebook).argsort()
    topk_bf_gain_magnitude_feedback_proposed_codebook, topk_acc_magnitude_feedback_proposed_codebook = topk_metrics_from_sorted(topk_sorted_test_magnitude_feedback_proposed_codebook, soft_label[test_idc,:], noise_power, M)
    magnitude_feedback_proposed_codebook_topk_gain.append(topk_bf_gain_magnitude_feedback_proposed_codebook)
    magnitude_feedback_proposed_codebook_topk_acc.append(topk_acc_magnitude_feedback_proposed_codebook)

    norandom_noise_key = str(float(noise_power_dBm))
    norandom_codebook_path = './Saved Codebooks/norandomphase_magnitude_feedback_proposed_codebook_{}_{}_beam_probing_codebook_{}_beam_noise_{}_dBm_tx_power_{}_dBm.npy'.format(dataset_name,n_wb,n_nb,norandom_noise_key,tx_power_dBm)
    norandom_model_path = './Saved Projection Models/norandomphase_magnitude_feedback_projection_trainer_{}_{}_beam_probing_codebook_{}_beam_noise_{}_dBm_tx_power_{}_dBm.pt'.format(dataset_name,n_wb,n_nb,norandom_noise_key,tx_power_dBm)
    if os.path.exists(norandom_codebook_path) and os.path.exists(norandom_model_path):
        norandomphase_proposed_codebook = np.load(norandom_codebook_path)
        norandomphase_proposed_codebook_model = Beam_Classifier(n_antenna=n_antenna,n_wide_beam=n_wb,n_narrow_beam=n_nb,
                                            trainable_codebook=False,complex_codebook=norandomphase_proposed_codebook,
                                            noise_power=noise_power,norm_factor=norm_factor)
        norandomphase_projection_training_model = projection_trainer_fake(if_fake,n_wide_beam=n_wb,n_narrow_beam=n_nb,narrow_beam_CB=dft_nb_codebook,
                                            noise_power=noise_power,norm_factor=norm_factor)
        norandomphase_projection_training_model.load_state_dict(torch.load(norandom_model_path, map_location='cpu'))
        norandomphase_proposed_codebook_model.dense1.load_state_dict(norandomphase_projection_training_model.dense1.state_dict())
        norandomphase_proposed_codebook_model.dense2.load_state_dict(norandomphase_projection_training_model.dense2.state_dict())
        norandomphase_proposed_codebook_model.dense3.load_state_dict(norandomphase_projection_training_model.dense3.state_dict())
        norandomphase_test_loss,norandomphase_test_acc = np.nan, np.nan
        norandomphase_proposed_codebook_acc.append(norandomphase_test_acc)
        y_test_predict_norandomphase = predict_numpy(norandomphase_proposed_codebook_model, torch_x_test, batch_size, device)
        topk_sorted_test_norandomphase = (-y_test_predict_norandomphase).argsort()
        topk_bf_gain_norandomphase, topk_acc_norandomphase = topk_metrics_from_sorted(topk_sorted_test_norandomphase, soft_label[test_idc,:], noise_power, M)
        norandomphase_proposed_codebook_topk_gain.append(topk_bf_gain_norandomphase)
        norandomphase_proposed_codebook_topk_acc.append(topk_acc_norandomphase)
    else:
        norandomphase_proposed_codebook_acc.append(np.nan)
        norandomphase_proposed_codebook_topk_gain.append(np.full((len(test_idc),3), np.nan))
        norandomphase_proposed_codebook_topk_acc.append(np.full(3, np.nan))
    
    optimal_gains.append(soft_label[test_idc,:].max(axis=-1))

    
dft_codebook_topk_gain = np.array(dft_codebook_topk_gain)
dft_prune_codebook_topk_gain = np.array(dft_prune_codebook_topk_gain)
learned_codebook_topk_gain = np.array(learned_codebook_topk_gain)
complex_feedback_codebook_topk_gain = np.array(complex_feedback_proposed_codebook_topk_gain)
magnitude_feedback_proposed_codebook_topk_gain = np.array(magnitude_feedback_proposed_codebook_topk_gain)
norandomphase_proposed_codebook_topk_gain = np.array(norandomphase_proposed_codebook_topk_gain)

dft_codebook_topk_snr = dft_codebook_topk_gain
dft_prune_codebook_topk_snr = dft_prune_codebook_topk_gain
learned_codebook_topk_snr = learned_codebook_topk_gain
complex_feedback_proposed_codebook_topk_snr = complex_feedback_codebook_topk_gain
magnitude_feedback_proposed_codebook_topk_snr = magnitude_feedback_proposed_codebook_topk_gain
norandomphase_proposed_codebook_topk_snr = norandomphase_proposed_codebook_topk_gain

optimal_gains = np.array(optimal_gains)


optimal_snr = tx_power_dBm + 10*np.log10(optimal_gains) - noise_power_dBm + noise_factor

for noise_power_dBm_ind, noise_power_dBm in enumerate(tqdm(noise_power_dBm_list, desc=f"{dataset_name} noise sweep")):
    dft_codebook_topk_snr[noise_power_dBm_ind,:,:] = tx_power_dBm + 10*np.log10(dft_codebook_topk_snr[noise_power_dBm_ind,:,:]) - noise_power_dBm + noise_factor
    dft_prune_codebook_topk_snr[noise_power_dBm_ind,:,:] = tx_power_dBm + 10*np.log10(dft_prune_codebook_topk_snr[noise_power_dBm_ind,:,:]) - noise_power_dBm + noise_factor    
    learned_codebook_topk_snr[noise_power_dBm_ind,:,:] = tx_power_dBm + 10*np.log10(learned_codebook_topk_snr[noise_power_dBm_ind,:,:]) - noise_power_dBm + noise_factor
    complex_feedback_proposed_codebook_topk_snr[noise_power_dBm_ind,:,:] = tx_power_dBm + 10*np.log10(complex_feedback_proposed_codebook_topk_snr[noise_power_dBm_ind,:,:]) - noise_power_dBm + noise_factor
    magnitude_feedback_proposed_codebook_topk_snr[noise_power_dBm_ind,:,:] = tx_power_dBm + 10*np.log10(magnitude_feedback_proposed_codebook_topk_snr[noise_power_dBm_ind,:,:]) - noise_power_dBm + noise_factor
    norandomphase_proposed_codebook_topk_snr[noise_power_dBm_ind,:,:] = tx_power_dBm + 10*np.log10(norandomphase_proposed_codebook_topk_snr[noise_power_dBm_ind,:,:]) - noise_power_dBm + noise_factor


dft_codebook_topk_acc = np.array(dft_codebook_topk_acc)
dft_prune_codebook_topk_acc = np.array(dft_prune_codebook_topk_acc)
learnable_codebook_topk_acc = np.array(learnable_codebook_topk_acc)
complex_feedback_proposed_codebook_topk_acc = np.array(complex_feedback_proposed_codebook_topk_acc)
magnitude_feedback_proposed_codebook_topk_acc = np.array(magnitude_feedback_proposed_codebook_topk_acc)
norandomphase_proposed_codebook_topk_acc = np.array(norandomphase_proposed_codebook_topk_acc)

#%%
"""    
compute exhaustive beam search acc and snr
"""

use_noisy_exhaustive = True

noiseless_exhaustive_avgsnr_all = []
noiseless_exhaustive_acc_all = []
exhaustive_acc_all=[]
exhaustive_avgsnr_all=[]

narrow_probing_beam_gain = []
ethan_probing_beam_gain = []
proposed_probing_beam_gain = []

for use_noisy_exhaustive in [False,True]:
    acc_list = exhaustive_acc_all if use_noisy_exhaustive else noiseless_exhaustive_acc_all
    avgsnr_list = exhaustive_avgsnr_all if use_noisy_exhaustive else noiseless_exhaustive_avgsnr_all

    for noise_power_dBm_ind, noise_power_dBm in enumerate(tqdm(noise_power_dBm_list, desc=f"{dataset_name} noise sweep")):
        noise_power = 10**((noise_power_dBm-tx_power_dBm-noise_factor)/10)
        dft_nb_codebook = DFT_codebook(nseg=64,n_antenna=64) # (nseg, n_antenna)
        dft_nb_az = DFT_angles(64)
        dft_nb_az = np.arcsin(dft_nb_az)
        nb_bf_signal = np.matmul(h[test_idc], dft_nb_codebook.conj().T)
        nb_bf_noise_real = np.random.normal(loc=0,scale=1,size=nb_bf_signal.shape)*np.sqrt(noise_power/2)
        nb_bf_noise_imag = np.random.normal(loc=0,scale=1,size=nb_bf_signal.shape)*np.sqrt(noise_power/2)
        nb_bf_signal_with_noise = nb_bf_signal + nb_bf_noise_real + 1j*nb_bf_noise_imag if use_noisy_exhaustive else nb_bf_signal
        nb_bf_gain = np.power(np.absolute(nb_bf_signal),2)
        nb_bf_gain_with_noise = np.power(np.absolute(nb_bf_signal_with_noise),2)


        # dft_nb_codebook.shape: (12, 64) nseg, num_antenna
        # magnitude_feedback_proposed_codebook.shape, learnable_codebook.shape # (12, 64), (64, 12)
        # h[test_idc].shape: (num_data, num_antenna)

        ethan_bf_power = np.power(np.absolute(np.matmul(h[test_idc], learnable_codebook)), 2)
        proposed_bf_power = np.power(np.absolute(np.matmul(h[test_idc], magnitude_feedback_proposed_codebook.T)), 2) # conjugate를 해야 하는지 확실하지 않음. 확인 필요

        narrow_probing_beam_gain.append(nb_bf_gain)
        ethan_probing_beam_gain.append(ethan_bf_power)
        proposed_probing_beam_gain.append(proposed_bf_power)

        best_nb_noisy = np.argmax(nb_bf_gain_with_noise,axis=1)
        best_nb = np.argmax(nb_bf_gain,axis=1)
        best_nb_az = dft_nb_az[best_nb]
        exhaustive_acc = (best_nb_noisy==best_nb).mean()
        acc_list.append(exhaustive_acc)
        genie_nb_snr = tx_power_dBm + 10*np.log10(nb_bf_gain.max(axis=1)) - noise_power_dBm + noise_factor
        exhaustive_nb_snr = np.array([nb_bf_gain[ue_idx,best_nb_idx_noisy] for ue_idx,best_nb_idx_noisy in enumerate(best_nb_noisy)])
        exhaustive_nb_snr = tx_power_dBm + 10*np.log10(exhaustive_nb_snr) - noise_power_dBm + noise_factor
        avgsnr_list.append(exhaustive_nb_snr.mean())
        print('Genie avg. SNR = {} dB.'.format(genie_nb_snr.mean()))
        print('Exhaustive accuracy = {}, avg. SNR = {} dB.'.format(exhaustive_acc,exhaustive_nb_snr.mean()))


narrow_probing_beam_gain = np.array(narrow_probing_beam_gain).flatten()
ethan_probing_beam_gain = np.array(ethan_probing_beam_gain).flatten()
proposed_probing_beam_gain = np.array(proposed_probing_beam_gain).flatten()

# print('Max:', 10 * np.log10(np.max(narrow_probing_beam_gain)), 'dB')
# print('Mean:', 10 * np.log10(np.mean(narrow_probing_beam_gain)), 'dB')

plt.hist(10 * np.log10(narrow_probing_beam_gain), bins='auto', density=True, histtype='step')
plt.hist(10 * np.log10(ethan_probing_beam_gain), bins='auto', density=True, histtype='step')
plt.hist(10 * np.log10(proposed_probing_beam_gain), bins='auto', density=True, histtype='step')

plt.title('Probing Beam Gain Histogram')
plt.legend(
    ['narrow beam', 'Learned w/ CSI', 'Proposed codebook']
)

plt.title('Histogram of Narrow Beam Gain (dB)')
plt.xlabel('Probing Beam Gain (dB)')
plt.show()

print('Narrow mean:', 10 * np.log10(np.mean(narrow_probing_beam_gain)), 'dB')
print('Ethan mean:', 10 * np.log10(np.mean(ethan_probing_beam_gain)), 'dB')
print('Proposed mean:', 10 * np.log10(np.mean(proposed_probing_beam_gain)), 'dB')


exhaustive_acc_all=np.array(exhaustive_acc_all)
exhaustive_avgsnr_all=np.array(exhaustive_avgsnr_all)

noiseless_exhaustive_acc_all=np.array(noiseless_exhaustive_acc_all)
noiseless_exhaustive_avgsnr_all=np.array(noiseless_exhaustive_avgsnr_all)
     
two_tier_AMCF_acc_all = []
two_tier_AMCF_avgsnr_all = []
n_wb=12
for noise_power_dBm_ind, noise_power_dBm in enumerate(tqdm(noise_power_dBm_list, desc=f"{dataset_name} noise sweep")):
    noise_power = 10**((noise_power_dBm-tx_power_dBm-noise_factor)/10)
    AMCF_wb_codebook = get_AMCF_codebook(n_wb, n_antenna)
    AMCF_wb_cv = AMCF_boundaries(n_wb)
    #handle numerical issue where theta is not exactly +-1
    AMCF_wb_cv[AMCF_wb_cv>1.0]=1.0
    AMCF_wb_cv[AMCF_wb_cv<-1.0]=-1.0
    AMCF_wb_cv = np.arcsin(AMCF_wb_cv)
    AMCF_wb_cv = np.flipud(AMCF_wb_cv)
    
    wb_2_nb = {}
    for bi in range(n_wb):
        children_nb = ((dft_nb_az>=AMCF_wb_cv[bi,0]) & (dft_nb_az<=AMCF_wb_cv[bi,1])).nonzero()[0]
        wb_2_nb[bi] = children_nb
        
    wb_bf_signal = np.matmul(h[test_idc], AMCF_wb_codebook.conj().T)
    wb_bf_noise_real = np.random.normal(loc=0,scale=1,size=wb_bf_signal.shape)*np.sqrt(noise_power/2)
    wb_bf_noise_imag = np.random.normal(loc=0,scale=1,size=wb_bf_signal.shape)*np.sqrt(noise_power/2)
    wb_bf_signal_with_noise = wb_bf_signal + wb_bf_noise_real + 1j*wb_bf_noise_imag
    wb_bf_gain = np.power(np.absolute(wb_bf_signal),2)
    wb_bf_gain_with_noise = np.power(np.absolute(wb_bf_signal_with_noise),2)
    best_wb_noisy = np.argmax(wb_bf_gain_with_noise,axis=1)
    
    nb_bf_signal = np.matmul(h[test_idc], dft_nb_codebook.conj().T)
    nb_bf_noise_real = np.random.normal(loc=0,scale=1,size=nb_bf_signal.shape)*np.sqrt(noise_power/2)
    nb_bf_noise_imag = np.random.normal(loc=0,scale=1,size=nb_bf_signal.shape)*np.sqrt(noise_power/2)
    nb_bf_signal_with_noise = nb_bf_signal + nb_bf_noise_real + 1j*nb_bf_noise_imag
    nb_bf_gain = np.power(np.absolute(nb_bf_signal),2)
    nb_bf_gain_with_noise = np.power(np.absolute(nb_bf_signal_with_noise),2)
    two_tier_best_nb = []
    for ue_idx,best_wb_idx in enumerate(best_wb_noisy):
        child_beam_idc = wb_2_nb[best_wb_idx]
        child_beam_gain_noisy = nb_bf_gain_with_noise[ue_idx,child_beam_idc]
        two_tier_best_nb.append(child_beam_idc[np.argmax(child_beam_gain_noisy)])
    two_tier_best_nb = np.array(two_tier_best_nb)    
    hierarchical_acc = (two_tier_best_nb==best_nb).mean() 
    
    two_tier_AMCF_acc_all.append(hierarchical_acc)
    
    best_wb_best_child_nb_snr = np.array([nb_bf_gain[ue_idx,two_tier_best_nb_idx] for ue_idx,two_tier_best_nb_idx in enumerate(two_tier_best_nb)])
    best_wb_best_child_nb_snr = tx_power_dBm + 10*np.log10(best_wb_best_child_nb_snr) - noise_power_dBm + noise_factor
    two_tier_AMCF_avgsnr_all.append(best_wb_best_child_nb_snr)

two_tier_AMCF_acc_all=np.array(two_tier_AMCF_acc_all)
two_tier_AMCF_avgsnr_all=np.mean(np.array(two_tier_AMCF_avgsnr_all),axis=1)



#%% ALL- ACC
#ALL - top3

# --- Accuracy vs. measurement noise power (dBm) ---  (same style: 1x2, single legend below)

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(6, 4), sharey=True, constrained_layout=True)

def plot_acc_noise(ax, k_idx, title):
    styles = {
        "Learned w/ CSI": dict(ls='-',  marker='o'),
        "Proposed":       dict(ls='--', marker='s'),
        "No phase aug.": dict(ls='--', marker='x'),
        "2-tier Hier.":   dict(ls='-.', marker='^'),
        "DFT classifier": dict(ls=':',  marker='D'),
        "DFT Pruning":    dict(ls=(0,(3,1,1,1)), marker='v'),
        "Noiseless Exh.": dict(ls='-',  marker=None),
        "Noisy Exh.":     dict(ls='--', marker=None),
    }

    # Exhaustive (arrays are already over noise)
    ax.plot(noise_power_dBm_list, noiseless_exhaustive_acc_all,
            label='Noiseless Exhaustive', color='tab:gray', **styles["Noiseless Exh."])
    ax.plot(noise_power_dBm_list, exhaustive_acc_all,
            label='Noisy Exhaustive', color='tab:gray', **styles["Noisy Exh."])

    # Methods
    ax.plot(noise_power_dBm_list, learnable_codebook_topk_acc[:, k_idx],
            label='Learned w/ CSI', **styles["Learned w/ CSI"])
    ax.plot(noise_power_dBm_list, magnitude_feedback_proposed_codebook_topk_acc[:, k_idx],
            label='Proposed', **styles["Proposed"])
    ax.plot(noise_power_dBm_list, norandomphase_proposed_codebook_topk_acc[:, k_idx],
            label='No phase aug.', **styles["No phase aug."])
    ax.plot(noise_power_dBm_list, two_tier_AMCF_acc_all,
            label='2-tier Hier.', **styles["2-tier Hier."])
    ax.plot(noise_power_dBm_list, dft_codebook_topk_acc[:, k_idx],
            label='DFT classifier', **styles["DFT classifier"])
    ax.plot(noise_power_dBm_list, dft_prune_codebook_topk_acc[:, k_idx],
            label='DFT Pruning', **styles["DFT Pruning"])

    ax.set_title(title)
    ax.set_xlabel('Measurement noise power (dBm)')
    ax.set_xticks(noise_power_dBm_list)
    ax.grid(True, alpha=0.3)
    # ax.set_ylim(0, None)

plot_acc_noise(axes[0], 0, "k = 1")
plot_acc_noise(axes[1], 2, "k = 3")   # k=3이 없으면 이 줄을 지우고, 위 plot_acc_noise도 k_idx=0만 쓰면 됨
axes[0].set_ylabel('Accuracy of top-k prediction')

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels,
           loc='lower center',
           bbox_to_anchor=(0.5, -0.05),
           ncol=3,
           frameon=True)

fig.tight_layout(rect=(0, 0.12, 1, 1))
plt.savefig(f'./fig/Noise_vs_Acc_{dataset_name}_{n_wb}_{n_nb}_{tx_power_dBm}_dBm.pdf', bbox_inches="tight", pad_inches=0.08)
plt.show()

Noise_vs_acc = []
Noise_vs_acc.append(learnable_codebook_topk_acc[:,2])
Noise_vs_acc.append(learnable_codebook_topk_acc[:,0])
Noise_vs_acc.append(magnitude_feedback_proposed_codebook_topk_acc[:,2])
Noise_vs_acc.append(magnitude_feedback_proposed_codebook_topk_acc[:,0])
Noise_vs_acc.append(exhaustive_acc_all)
Noise_vs_acc.append(two_tier_AMCF_acc_all)
Noise_vs_acc.append(noiseless_exhaustive_acc_all)
# savemat('./Saved matfile/Noise_vs_acc_{}_{}_beam_probing_codebook_{}_beam_tx_power_{}_dBm.mat'.format(dataset_name,n_wb,n_nb,tx_power_dBm), {'Noise_vs_acc':Noise_vs_acc})

#%% ALL - SNR

# --- Average SNR vs. measurement noise power (dBm) ---  (same style: 1x2, single legend below)

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(6, 4), sharey=True, constrained_layout=True)

def plot_snr_noise(ax, k_idx, title):
    styles = {
        "Learned w/ CSI": dict(ls='-',  marker='o'),
        "Proposed":       dict(ls='--', marker='s'),
        "No phase aug.": dict(ls='--', marker='x'),
        "2-tier Hier.":   dict(ls='-.', marker='^'),
        "DFT classifier": dict(ls=':',  marker='D'),
        "DFT Pruning":    dict(ls=(0,(3,1,1,1)), marker='v'),
        "Noiseless Exh.": dict(ls='-', marker=None),
        "Noisy Exh.":     dict(ls='--',  marker=None),
    }

    # Exhaustive (arrays are already over noise)
    ax.plot(noise_power_dBm_list, noiseless_exhaustive_avgsnr_all,
            label='Noiseless Exhaustive', color='tab:gray', **styles["Noiseless Exh."])
    ax.plot(noise_power_dBm_list, exhaustive_avgsnr_all,
            label='Noisy Exhaustive', color='tab:gray', **styles["Noisy Exh."])

    # Methods
    ax.plot(noise_power_dBm_list, learned_codebook_topk_snr[:, :, k_idx].mean(axis=1),
            label='Learned w/ CSI', **styles["Learned w/ CSI"])
    ax.plot(noise_power_dBm_list, magnitude_feedback_proposed_codebook_topk_snr[:, :, k_idx].mean(axis=1),
            label='Proposed', **styles["Proposed"])
    ax.plot(noise_power_dBm_list, norandomphase_proposed_codebook_topk_snr[:, :, k_idx].mean(axis=1),
            label='No phase aug.', **styles["No phase aug."])
    ax.plot(noise_power_dBm_list, two_tier_AMCF_avgsnr_all,
            label='2-tier Hier.', **styles["2-tier Hier."])
    ax.plot(noise_power_dBm_list, dft_codebook_topk_snr[:, :, k_idx].mean(axis=1),
            label='DFT classifier', **styles["DFT classifier"])
    ax.plot(noise_power_dBm_list, dft_prune_codebook_topk_snr[:, :, k_idx].mean(axis=1),
            label='DFT Pruning', **styles["DFT Pruning"])

    ax.set_title(title)
    ax.set_xlabel('Measurement noise power (dBm)')
    ax.set_xticks(noise_power_dBm_list)
    ax.grid(True, alpha=0.3)

plot_snr_noise(axes[0], 0, "k = 1")
plot_snr_noise(axes[1], 2, "k = 3")   # k=3이 없으면 이 줄을 지우고, 위 plot_snr_noise도 k_idx=0만 쓰면 됨
axes[0].set_ylabel('Average SNR (dB)')

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels,
           loc='lower center',
           bbox_to_anchor=(0.5, -0.05),
           ncol=3,
           frameon=True)

fig.tight_layout(rect=(0, 0.12, 1, 1))
plt.savefig(f'./fig/Noise_vs_SNR_{dataset_name}_{n_wb}_{n_nb}_{tx_power_dBm}_dBm.pdf', bbox_inches="tight", pad_inches=0.08)
plt.show()

Noise_vs_snr = []
Noise_vs_snr.append(learned_codebook_topk_snr[:,:,2].mean(axis=1))
Noise_vs_snr.append(learned_codebook_topk_snr[:,:,0].mean(axis=1))
Noise_vs_snr.append(magnitude_feedback_proposed_codebook_topk_snr[:,:,2].mean(axis=1))
Noise_vs_snr.append(magnitude_feedback_proposed_codebook_topk_snr[:,:,0].mean(axis=1))
Noise_vs_snr.append(exhaustive_avgsnr_all)
Noise_vs_snr.append(two_tier_AMCF_avgsnr_all)
Noise_vs_snr.append(noiseless_exhaustive_avgsnr_all)
# savemat('./Saved matfile/Noise_vs_snr_{}_{}_beam_probing_codebook_{}_beam_tx_power_{}_dBm.mat'.format(dataset_name,n_wb,n_nb,tx_power_dBm), {'Noise_vs_snr':Noise_vs_snr})
#%%
# Exhaustive search-based pruning

noise_power_dBm = -81
n_wb = 20
n_nb = 64
noise_power = 10**((noise_power_dBm-tx_power_dBm-noise_factor)/10)

dft_wb_codebook = DFT_codebook(nseg=n_wb,n_antenna=n_antenna)

dft_codebook_model = Beam_Classifier(n_antenna=n_antenna,n_wide_beam=n_wb,n_narrow_beam=n_nb,
                                        trainable_codebook=False,complex_codebook=dft_wb_codebook,
                                        noise_power=noise_power,norm_factor=norm_factor)

dft_model_savefname = './Saved Models/DFTprob_{}_{}_beam_probing_codebook_{}_beam_noise_{}_dBm_tx_power_{}_dBm.pt'.format(dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm)
dft_codebook_model.load_state_dict(torch.load(dft_model_savefname))

least_indicies = []
prune_accs = []
dft_wb_codebook_ = DFT_codebook(nseg=n_wb,n_antenna=n_antenna)

progress = tqdm(total=n_wb*10*n_wb)
for _ in range(n_wb):
    t = []
    for _ in range(10):
        for i in range(n_wb):
            if i in least_indicies:
                t.append(0.)
                progress.update(1)
                continue

            dft_wb_codebook = dft_wb_codebook_.copy()
            dft_wb_codebook[i] = np.zeros_like(dft_wb_codebook[i])

            dft_codebook_model.complex_codebook = dft_wb_codebook # n_beams x n_antenna
            cb_blockmatrix = codebook_blockmatrix(dft_codebook_model.complex_codebook.T)
            dft_codebook_model.codebook = torch.from_numpy(cb_blockmatrix).float()
            dft_codebook_model.codebook.requires_grad = False

            dft_codebook_test_loss,dft_codebook_test_acc = np.nan, np.nan 
            t.append(dft_codebook_test_acc)
            progress.update(1)

    t = np.reshape(np.array(t), (-1, n_wb))
    t = np.mean(t, axis=0)

    for j in least_indicies:
        t[j] = 0.

    least_idx = np.argmax(t)
    least_indicies.append(least_idx)
    prune_accs.append(t[least_idx])
    dft_wb_codebook_[least_idx] = np.zeros_like(dft_wb_codebook_[0])

# %%
print(dataset_name, least_indicies)
# Rosslyn_ULA [7, 6, 8, 9, 10, 12, 11, 13, 3, 5, 4, 2, 14, 1, 15, 16, 17, 18, 19, 0]
# O1_28B_ULA [9, 13, 5, 12, 7, 11, 3, 6, 14, 4, 8, 15, 10, 16, 2, 17, 1, 18, 0, 19]

plt.plot(prune_accs, 'x-')
# %%
