# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 20:04:13 2021

@author: Yuqiang (Ethan) Heng
"""
#%%
# os.chdir('/root/bkup/Learning_site_specific')

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from ComplexLayers_Torch import Beam_Classifier,Beam_Classifier_Quantization, eval_model
from ComplexLayers_Torch import Beam_Classifier, projection_trainer,projection_trainer_fake, fit,fit_projector,  eval_model
import torch.utils.data
import torch.nn as nn
from sklearn.model_selection import train_test_split
from scipy.io import savemat

from beam_utils_L import ULA_DFT_codebook as DFT_codebook
from beam_utils_L import plot_codebook_pattern, DFT_angles, Beam_Search_Tree, get_AMCF_codebook, AMCF_boundaries
from beam_utils_L import codebook_blockmatrix
np.random.seed(7)
# number of probing beams (N_W)
n_wide_beams = [16]
# number of narrow data beams to select from (N_V)
n_narrow_beams = [64 for i in n_wide_beams]
n_antenna = 64
batch_size = 500
noise_power_dBm = -81
noise_factor = 0 #dB
nb_topk_list=list(np.concatenate(([1,2],np.arange(4,129,4))))
dB_stepsize_list=list(np.arange(0,21,2))

noiseless = False
 
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
noise_power = 10**((noise_power_dBm-tx_power_dBm-noise_factor)/10)
top_k=1

noise_power = 10**((noise_power_dBm-tx_power_dBm-noise_factor)/10)
norm_factor = np.max(abs(h))
h_scaled = h/norm_factor
h_concat_scaled = np.concatenate((h_real/norm_factor,h_imag/norm_factor),axis=1)

train_idc, test_idc = train_test_split(np.arange(h.shape[0]),test_size=0.4)
val_idc, test_idc = train_test_split(test_idc,test_size=0.5)



dft_codebook_acc = []
learnable_codebook_acc = []
complex_feedback_proposed_codebook_acc = []
magnitude_feedback_proposed_codebook_acc = []

dft_codebook_topk_acc = []
learnable_codebook_topk_acc = []
complex_feedback_proposed_codebook_topk_acc = []
magnitude_feedback_proposed_codebook_topk_acc=[]

dft_codebook_topk_gain = []
learned_codebook_topk_gain= []
complex_feedback_proposed_codebook_topk_gain= []
magnitude_feedback_proposed_codebook_topk_gain = []

for nb_topk_ind,nb_topk in enumerate(nb_topk_list):
    print('nb_topk={}'.format(nb_topk))
    for db_stepsize_ind, dB_stepsize in enumerate(dB_stepsize_list):
        print('stepsizedB={}'.format(dB_stepsize))
        for n_wb_i, n_wb in enumerate(n_wide_beams):
            n_nb = n_narrow_beams[n_wb_i]
            print('{} Wide Beams, {} Narrow Beams.'.format(n_wb,n_nb))
            dB_stepsize_W=dB_stepsize




            optimal_gains = []

            

            learnable_codebook =np.load('./Saved Codebooks/Ethan_{}_{}_beam_probing_codebook_{}_beam_noise_{}_dBm_tx_power_{}_dBm.npy'.format(dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm))
            # complex_feedback_proposed_codebook = np.load('./Saved Codebooks/lowcomplexity_top{}_{}dBstepsize_complex_feedback_proposed_codebook_{}_{}_beam_probing_codebook_{}_beam_noise_{}_dBm_tx_power_{}_dBm.npy'.format(nb_topk,dB_stepsize,dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm))
            magnitude_feedback_proposed_codebook = np.load('./Saved Codebooks/lowcomplexity_top{}_{}dBstepsize_magnitude_feedback_proposed_codebook_{}_{}_beam_probing_codebook_{}_beam_noise_{}_dBm_tx_power_{}_dBm.npy'.format(nb_topk,dB_stepsize,dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm))

            
                
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

            learnable_codebook_test_loss,learnable_codebook_test_acc = eval_model(learnable_codebook_model,test_loader,nn.CrossEntropyLoss()) 
            learnable_codebook_acc.append(learnable_codebook_test_acc)
            y_test_predict_learnable_codebook = learnable_codebook_model(torch_x_test.float()).detach().numpy()
            topk_sorted_test_learned_codebook = (-y_test_predict_learnable_codebook).argsort()
            topk_bf_gain_learnable_codebook = []
            topk_acc_learnable_codebook = []
            for ue_bf_gain, pred_sort in zip(soft_label[test_idc,:],topk_sorted_test_learned_codebook):
                topk_gains = [ue_bf_gain[pred_sort[:k]].max() for k in range(1,4)]
                topk_bf_gain_learnable_codebook.append(topk_gains)
                topk_acc = [ue_bf_gain.argmax() in pred_sort[:k] for k in range(1,4)]
                topk_acc_learnable_codebook.append(topk_acc)
            
            topk_bf_gain_learnable_codebook = np.array(topk_bf_gain_learnable_codebook)
            learned_codebook_topk_gain.append(topk_bf_gain_learnable_codebook)
            topk_acc_learnable_codebook = np.array(topk_acc_learnable_codebook).mean(axis=0)
            learnable_codebook_topk_acc.append(topk_acc_learnable_codebook)

            
            dft_wb_codebook = DFT_codebook(nseg=n_wb,n_antenna=n_antenna)
            dft_codebook_model = Beam_Classifier(n_antenna=n_antenna,n_wide_beam=n_wb,n_narrow_beam=n_nb,
                                                trainable_codebook=False,complex_codebook=dft_wb_codebook,
                                                noise_power=noise_power,norm_factor=norm_factor)
            dft_model_savefname = './Saved Models/DFTprob_{}_{}_beam_probing_codebook_{}_beam_noise_{}_dBm_tx_power_{}_dBm.pt'.format(dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm)
            dft_codebook_model.load_state_dict(torch.load(dft_model_savefname))
            dft_codebook_test_loss,dft_codebook_test_acc = eval_model(dft_codebook_model,test_loader,nn.CrossEntropyLoss()) 
            dft_codebook_acc.append(dft_codebook_test_acc)
            y_test_predict_dft_codebook = dft_codebook_model(torch_x_test.float()).detach().numpy()
            topk_sorted_test_dft_codebook = (-y_test_predict_dft_codebook).argsort()
            topk_bf_gain_dft_codebook = []
            topk_acc_dft_codebook = []
            for ue_bf_gain, pred_sort in zip(soft_label[test_idc,:],topk_sorted_test_dft_codebook):
                topk_gains = [ue_bf_gain[pred_sort[:k]].max() for k in range(1,4)]
                topk_bf_gain_dft_codebook.append(topk_gains)
                topk_acc = [ue_bf_gain.argmax() in pred_sort[:k] for k in range(1,4)]
                topk_acc_dft_codebook.append(topk_acc)        
            topk_bf_gain_dft_codebook = np.array(topk_bf_gain_dft_codebook)
            dft_codebook_topk_gain.append(topk_bf_gain_dft_codebook)
            topk_acc_dft_codebook = np.array(topk_acc_dft_codebook).mean(axis=0)
            dft_codebook_topk_acc.append(topk_acc_dft_codebook)

            

            # complex_feedback_proposed_codebook_model = Beam_Classifier_Quantization(dB_stepsize_W,n_antenna=n_antenna,n_wide_beam=n_wb,n_narrow_beam=n_nb,
            #                                     trainable_codebook=False,complex_codebook=complex_feedback_proposed_codebook,
            #                                     noise_power=noise_power,norm_factor=norm_factor)
            # complex_feedback_proposed_model_savefname = './Saved Models/lowcomplexity_top{}_{}dBstepsize_complex_feedback_propsed_{}_{}_beam_probing_codebook_{}_beam_noise_{}_dBm_tx_power_{}_dBm.pt'.format(nb_topk,dB_stepsize,dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm)
            # # complex_feedback_proposed_codebook_model.load_state_dict(torch.load(complex_feedback_proposed_model_savefname))

            # complex_feedback_projection_training_model = projection_trainer(n_wide_beam=n_wb,n_narrow_beam=n_nb,narrow_beam_CB=dft_nb_codebook,
            #                                             noise_power=noise_power,norm_factor=norm_factor)
            # complex_feedback_projection_training_model_savefname = './Saved Projection Models/lowcomplexity_top{}_{}dBstepsize_complex_feedback_projection_trainer_{}_{}_beam_probing_codebook_{}_beam_noise_{}_dBm_tx_power_{}_dBm.pt'.format(nb_topk,dB_stepsize,dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm)
            # complex_feedback_projection_training_model.load_state_dict(torch.load(complex_feedback_projection_training_model_savefname))
            
            # complex_feedback_proposed_codebook_model.dense1.load_state_dict(complex_feedback_projection_training_model.dense1.state_dict())
            # complex_feedback_proposed_codebook_model.dense2.load_state_dict(complex_feedback_projection_training_model.dense2.state_dict())
            # complex_feedback_proposed_codebook_model.dense3.load_state_dict(complex_feedback_projection_training_model.dense3.state_dict())

            # complex_feedback_proposed_codebook_test_loss,complex_feedback_proposed_codebook_test_acc = eval_model(complex_feedback_proposed_codebook_model,test_loader,nn.CrossEntropyLoss()) 
            # complex_feedback_proposed_codebook_acc.append(complex_feedback_proposed_codebook_test_acc)
            # y_test_predict_complex_feedback_proposed_codebook = complex_feedback_proposed_codebook_model(torch_x_test.float()).detach().numpy()
            # topk_sorted_test_complex_feedback_proposed_codebook = (-y_test_predict_complex_feedback_proposed_codebook).argsort()
            # topk_bf_gain_complex_feedback_proposed_codebook = []
            # topk_acc_complex_feedback_proposed_codebook = []
            # for ue_bf_gain, pred_sort in zip(soft_label[test_idc,:],topk_sorted_test_complex_feedback_proposed_codebook):
            #     topk_gains = [ue_bf_gain[pred_sort[:k]].max() for k in range(1,4)]
            #     topk_bf_gain_complex_feedback_proposed_codebook.append(topk_gains)
            #     topk_acc = [ue_bf_gain.argmax() in pred_sort[:k] for k in range(1,4)]
            #     topk_acc_complex_feedback_proposed_codebook.append(topk_acc)
            # topk_bf_gain_complex_feedback_proposed_codebook = np.array(topk_bf_gain_complex_feedback_proposed_codebook)
            # complex_feedback_proposed_codebook_topk_gain.append(topk_bf_gain_complex_feedback_proposed_codebook)
            # topk_acc_complex_feedback_proposed_codebook = np.array(topk_acc_complex_feedback_proposed_codebook).mean(axis=0)
            # complex_feedback_proposed_codebook_topk_acc.append(topk_acc_complex_feedback_proposed_codebook)
            
            magnitude_feedback_proposed_codebook_model = Beam_Classifier_Quantization(dB_stepsize_W,n_antenna=n_antenna,n_wide_beam=n_wb,n_narrow_beam=n_nb,
                                                trainable_codebook=False,complex_codebook=magnitude_feedback_proposed_codebook,
                                                noise_power=noise_power,norm_factor=norm_factor)
            magnitude_feedback_proposed_codebook_model_savefname = './Saved Models/lowcomplexity_top{}_{}dBstepsize_magnitude_feedback_propsed_{}_{}_beam_probing_codebook_{}_beam_noise_{}_dBm_tx_power_{}_dBm.pt'.format(nb_topk,dB_stepsize,dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm)
            # magnitude_feedback_proposed_codebook_model.load_state_dict(torch.load(magnitude_feedback_proposed_codebook_model_savefname))
            
            
            if_fake=True
            magnitude_feedback_projection_training_model = projection_trainer_fake(if_fake,n_wide_beam=n_wb,n_narrow_beam=n_nb,narrow_beam_CB=dft_nb_codebook,
                                                        noise_power=noise_power,norm_factor=norm_factor)
            magnitude_feedback_projection_training_model_savefname = './Saved Projection Models/lowcomplexity_top{}_{}dBstepsize_magnitude_feedback_projection_trainer_{}_{}_beam_probing_codebook_{}_beam_noise_{}_dBm_tx_power_{}_dBm.pt'.format(nb_topk,dB_stepsize,dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm)
            magnitude_feedback_projection_training_model.load_state_dict(torch.load(magnitude_feedback_projection_training_model_savefname))
            
            magnitude_feedback_proposed_codebook_model.dense1.load_state_dict(magnitude_feedback_projection_training_model.dense1.state_dict())
            magnitude_feedback_proposed_codebook_model.dense2.load_state_dict(magnitude_feedback_projection_training_model.dense2.state_dict())
            magnitude_feedback_proposed_codebook_model.dense3.load_state_dict(magnitude_feedback_projection_training_model.dense3.state_dict())

            
            magnitude_feedback_proposed_codebook_test_loss,magnitude_feedback_proposed_codebook_test_acc = eval_model(magnitude_feedback_proposed_codebook_model,test_loader,nn.CrossEntropyLoss()) 
            magnitude_feedback_proposed_codebook_acc.append(magnitude_feedback_proposed_codebook_test_acc)
            y_test_predict_magnitude_feedback_proposed_codebook = magnitude_feedback_proposed_codebook_model(torch_x_test.float()).detach().numpy()
            topk_sorted_test_magnitude_feedback_proposed_codebook = (-y_test_predict_magnitude_feedback_proposed_codebook).argsort()
            topk_bf_gain_magnitude_feedback_proposed_codebook = []
            topk_acc_magnitude_feedback_proposed_codebook = []
            for ue_bf_gain, pred_sort in zip(soft_label[test_idc,:],topk_sorted_test_magnitude_feedback_proposed_codebook):
                topk_gains = [ue_bf_gain[pred_sort[:k]].max() for k in range(1,4)]
                topk_bf_gain_magnitude_feedback_proposed_codebook.append(topk_gains)
                topk_acc = [ue_bf_gain.argmax() in pred_sort[:k] for k in range(1,4)]
                topk_acc_magnitude_feedback_proposed_codebook.append(topk_acc)
            topk_bf_gain_magnitude_feedback_proposed_codebook = np.array(topk_bf_gain_magnitude_feedback_proposed_codebook)
            magnitude_feedback_proposed_codebook_topk_gain.append(topk_bf_gain_magnitude_feedback_proposed_codebook)
            topk_acc_magnitude_feedback_proposed_codebook = np.array(topk_acc_magnitude_feedback_proposed_codebook).mean(axis=0)
            magnitude_feedback_proposed_codebook_topk_acc.append(topk_acc_magnitude_feedback_proposed_codebook)
            
            optimal_gains.append(soft_label[test_idc,:].max(axis=-1))

dft_codebook_topk_gain = np.array(dft_codebook_topk_gain)
learned_codebook_topk_gain = np.array(learned_codebook_topk_gain)
# complex_feedback_codebook_topk_gain = np.array(complex_feedback_proposed_codebook_topk_gain)
magnitude_feedback_proposed_codebook_topk_gain = np.array(magnitude_feedback_proposed_codebook_topk_gain)

optimal_gains = np.array(optimal_gains)

dft_codebook_topk_snr = tx_power_dBm + 10*np.log10(dft_codebook_topk_gain) - noise_power_dBm + noise_factor
learned_codebook_topk_snr = tx_power_dBm + 10*np.log10(learned_codebook_topk_gain) - noise_power_dBm + noise_factor
# complex_feedback_proposed_codebook_topk_snr = tx_power_dBm + 10*np.log10(complex_feedback_proposed_codebook_topk_gain) - noise_power_dBm + noise_factor
magnitude_feedback_proposed_codebook_topk_snr = tx_power_dBm + 10*np.log10(magnitude_feedback_proposed_codebook_topk_gain) - noise_power_dBm + noise_factor

optimal_snr = tx_power_dBm + 10*np.log10(optimal_gains) - noise_power_dBm + noise_factor


dft_codebook_topk_acc = np.array(dft_codebook_topk_acc)
learnable_codebook_topk_acc = np.array(learnable_codebook_topk_acc)
# complex_feedback_proposed_codebook_topk_acc = np.array(complex_feedback_proposed_codebook_topk_acc)
magnitude_feedback_proposed_codebook_topk_acc = np.array(magnitude_feedback_proposed_codebook_topk_acc)



dft_codebook_topk_snr = np.reshape(dft_codebook_topk_snr,(len(nb_topk_list),len(dB_stepsize_list),dft_codebook_topk_snr.shape[-2],dft_codebook_topk_snr.shape[-1]))
learnable_codebook_topk_snr = np.reshape(learned_codebook_topk_snr,(len(nb_topk_list),len(dB_stepsize_list),learned_codebook_topk_snr.shape[-2],learned_codebook_topk_snr.shape[-1]))
# complex_feedback_proposed_codebook_topk_snr = np.reshape(complex_feedback_proposed_codebook_topk_snr,(len(nb_topk_list),len(dB_stepsize_list),complex_feedback_proposed_codebook_topk_snr.shape[-2],complex_feedback_proposed_codebook_topk_snr.shape[-1]))
magnitude_feedback_proposed_codebook_topk_snr = np.reshape(magnitude_feedback_proposed_codebook_topk_snr,(len(nb_topk_list),len(dB_stepsize_list),magnitude_feedback_proposed_codebook_topk_snr.shape[-2],magnitude_feedback_proposed_codebook_topk_snr.shape[-1]))



dft_codebook_topk_acc = np.reshape(dft_codebook_topk_acc,(len(nb_topk_list),len(dB_stepsize_list),dft_codebook_topk_acc.shape[-1]))
learnable_codebook_topk_acc = np.reshape(learnable_codebook_topk_acc,(len(nb_topk_list),len(dB_stepsize_list),learnable_codebook_topk_acc.shape[-1]))
# complex_feedback_proposed_codebook_topk_acc = np.reshape(complex_feedback_proposed_codebook_topk_acc,(len(nb_topk_list),len(dB_stepsize_list),complex_feedback_proposed_codebook_topk_acc.shape[-1]))
magnitude_feedback_proposed_codebook_topk_acc = np.reshape(magnitude_feedback_proposed_codebook_topk_acc,(len(nb_topk_list),len(dB_stepsize_list),magnitude_feedback_proposed_codebook_topk_acc.shape[-1]))


# savemat('./Saved matfile/magnitude_feedback_proposed_codebook_topk_acc_{}_{}_beam_probing_codebook_{}_beam_noise_{}_dBm_tx_power_{}_dBm.mat'.format(dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm), {'magnitude_feedback_proposed_codebook_topk_acc':magnitude_feedback_proposed_codebook_topk_acc})
# savemat('./Saved matfile/magnitude_feedback_proposed_codebook_topk_snr_{}_{}_beam_probing_codebook_{}_beam_noise_{}_dBm_tx_power_{}_dBm.mat'.format(dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm), {'magnitude_feedback_proposed_codebook_topk_snr':magnitude_feedback_proposed_codebook_topk_snr})

"""    
compute exhaustive beam search acc and snr
"""
dft_nb_codebook = DFT_codebook(nseg=64,n_antenna=64)
dft_nb_az = DFT_angles(64)
dft_nb_az = np.arcsin(1/0.5*dft_nb_az)
nb_bf_signal = np.matmul(h[test_idc], dft_nb_codebook.conj().T)
nb_bf_noise_real = np.random.normal(loc=0,scale=1,size=nb_bf_signal.shape)*np.sqrt(noise_power/2)
nb_bf_noise_imag = np.random.normal(loc=0,scale=1,size=nb_bf_signal.shape)*np.sqrt(noise_power/2)
nb_bf_signal_with_noise = nb_bf_signal + nb_bf_noise_real + 1j*nb_bf_noise_imag
nb_bf_gain = np.power(np.absolute(nb_bf_signal),2)
nb_bf_gain_with_noise = np.power(np.absolute(nb_bf_signal_with_noise),2)
best_nb_noisy = np.argmax(nb_bf_gain_with_noise,axis=1)
best_nb = np.argmax(nb_bf_gain,axis=1)
best_nb_az = dft_nb_az[best_nb]
exhaustive_acc = (best_nb_noisy==best_nb).mean()
genie_nb_snr = tx_power_dBm + 10*np.log10(nb_bf_gain.max(axis=1)) - noise_power_dBm + noise_factor
exhaustive_nb_snr = np.array([nb_bf_gain[ue_idx,best_nb_idx_noisy] for ue_idx,best_nb_idx_noisy in enumerate(best_nb_noisy)])
exhaustive_nb_snr = tx_power_dBm + 10*np.log10(exhaustive_nb_snr) - noise_power_dBm + noise_factor
print('Genie avg. SNR = {} dB.'.format(genie_nb_snr.mean()))
print('Exhaustive search accuracy = {}, avg. SNR = {} dB.'.format(exhaustive_acc,exhaustive_nb_snr.mean()))




    

#%%
# "Opt narrow beam prediction accuracy"

#Ethan
plt.figure(figsize=(6,4.5))
plt.plot(n_wide_beams,learnable_codebook_topk_acc[:,2],'--',marker='s',label='Ethan, k={}'.format(3))
plt.plot(n_wide_beams,learnable_codebook_topk_acc[:,1],'-.',marker='s',label='Ethan, k={}'.format(2))
plt.plot(n_wide_beams,learnable_codebook_topk_acc[:,0],marker='s',label='Ethan, k={}'.format(1))
plt.hlines(y=exhaustive_acc,xmin=min(n_wide_beams),xmax=max(n_wide_beams),linestyles='dotted',label='Exhaustive search')
plt.xticks(n_wide_beams)
plt.legend()
plt.xlabel('probing codebook size')
plt.ylabel('Accuracy')
plt.title('Optimal narrow beam prediction')
plt.show()


#DFT
plt.figure(figsize=(6,4.5))
plt.plot(n_wide_beams,dft_codebook_topk_acc[:,2],'--',marker='s',label='DFT, k={}'.format(3))
plt.plot(n_wide_beams,dft_codebook_topk_acc[:,1],'-.',marker='s',label='DFT, k={}'.format(2))
plt.plot(n_wide_beams,dft_codebook_topk_acc[:,0],marker='s',label='DFT, k={}'.format(1))
plt.hlines(y=exhaustive_acc,xmin=min(n_wide_beams),xmax=max(n_wide_beams),linestyles='dotted',label='Exhaustive search')
plt.xticks(n_wide_beams)
plt.legend()
plt.xlabel('probing codebook size')
plt.ylabel('Accuracy')
plt.title('Optimal narrow beam prediction accuracy')
plt.show()

#Proposed, complex
plt.figure(figsize=(6,4.5))
plt.plot(n_wide_beams,complex_feedback_proposed_codebook_topk_acc[:,2],'--',marker='s',label='Proposed method, complex feedback, k={}'.format(3))
plt.plot(n_wide_beams,complex_feedback_proposed_codebook_topk_acc[:,1],'-.',marker='s',label='Proposed method, complex feedback, k={}'.format(2))
plt.plot(n_wide_beams,complex_feedback_proposed_codebook_topk_acc[:,0],marker='s',label='Proposed method, complex feedback, k={}'.format(1))
plt.hlines(y=exhaustive_acc,xmin=min(n_wide_beams),xmax=max(n_wide_beams),linestyles='dotted',label='Exhaustive search')
plt.xticks(n_wide_beams)
plt.legend()
plt.xlabel('probing codebook size')
plt.ylabel('Accuracy')
plt.title('Optimal narrow beam prediction accuracy')
plt.show()

#Proposed, magnitude
plt.figure(figsize=(6,4.5))
plt.plot(n_wide_beams,magnitude_feedback_proposed_codebook_topk_acc[:,2],'--',marker='s',label='Proposed method, magnitude feedback, k={}'.format(3))
plt.plot(n_wide_beams,magnitude_feedback_proposed_codebook_topk_acc[:,1],'-.',marker='s',label='Proposed method, magnitude feedback, k={}'.format(2))
plt.plot(n_wide_beams,magnitude_feedback_proposed_codebook_topk_acc[:,0],marker='s',label='Proposed method, magnitude feedback, k={}'.format(1))
plt.hlines(y=exhaustive_acc,xmin=min(n_wide_beams),xmax=max(n_wide_beams),linestyles='dotted',label='Exhaustive search')
plt.xticks(n_wide_beams)
plt.legend()
plt.xlabel('probing codebook size')
plt.ylabel('Accuracy')
plt.title('Optimal narrow beam prediction accuracy')
plt.show()


"Average SNRs"


#Ethan
plt.figure(figsize=(6,4.5))
plt.plot(n_wide_beams,learned_codebook_topk_snr[:,:,2].mean(axis=1),'--',marker='s',label='Ethan, k={}'.format(3))
plt.plot(n_wide_beams,learned_codebook_topk_snr[:,:,1].mean(axis=1),'-.',marker='s',label='Ethan, k={}'.format(2))
plt.plot(n_wide_beams,learned_codebook_topk_snr[:,:,0].mean(axis=1),marker='s',label='Ethan, k={}'.format(1))
plt.xticks(n_wide_beams)
plt.legend()
plt.xlabel('probing codebook size')
plt.ylabel('Average SNR (dB)')
plt.title('SNR of top-k predicted beams')
plt.show()

#DFT
plt.figure(figsize=(6,4.5))
plt.plot(n_wide_beams,dft_codebook_topk_snr[:,:,2].mean(axis=1),'--',marker='s',label='DFT, k={}'.format(3))
plt.plot(n_wide_beams,dft_codebook_topk_snr[:,:,1].mean(axis=1),'-.',marker='s',label='DFT, k={}'.format(2))
plt.plot(n_wide_beams,dft_codebook_topk_snr[:,:,0].mean(axis=1),marker='s',label='DFT, k={}'.format(1))
plt.xticks(n_wide_beams)
plt.legend()
plt.xlabel('probing codebook size')
plt.ylabel('Average SNR (dB)')
plt.title('SNR of top-k predicted beams')
plt.show()

# #Proposed complex
# plt.figure(figsize=(6,4.5))
# plt.plot(n_wide_beams,complex_feedback_proposed_codebook_topk_snr[:,:,2].mean(axis=1),'--',marker='s',label='Proposed, complex feedback, k={}'.format(3))
# plt.plot(n_wide_beams,complex_feedback_proposed_codebook_topk_snr[:,:,1].mean(axis=1),'-.',marker='s',label='Proposed, complex feedback, k={}'.format(2))
# plt.plot(n_wide_beams,complex_feedback_proposed_codebook_topk_snr[:,:,0].mean(axis=1),marker='s',label='Proposed, complex feedback, k={}'.format(1))
# plt.xticks(n_wide_beams)
# plt.legend()
# plt.xlabel('probing codebook size')
# plt.ylabel('Average SNR (dB)')
# plt.title('SNR of top-k predicted beams')
# plt.show()



#Proposed magnitude
plt.figure(figsize=(6,4.5))
plt.plot(n_wide_beams,magnitude_feedback_proposed_codebook_topk_snr[:,:,2].mean(axis=1),'--',marker='s',label='Proposed, magnitude feedback, k={}'.format(3))
plt.plot(n_wide_beams,magnitude_feedback_proposed_codebook_topk_snr[:,:,1].mean(axis=1),'-.',marker='s',label='Proposed, magnitude feedback, k={}'.format(2))
plt.plot(n_wide_beams,magnitude_feedback_proposed_codebook_topk_snr[:,:,0].mean(axis=1),marker='s',label='Proposed, magnitude feedback, k={}'.format(1))
plt.xticks(n_wide_beams)
plt.legend()
plt.xlabel('probing codebook size')
plt.ylabel('Average SNR (dB)')
plt.title('SNR of top-k predicted beams')
plt.show()


#%%  Heatmap plt

# Create meshgrid for x and y coordinates
mesh_nbtopk = np.array(nb_topk_list)
mesh_dBstepsize = np.array(dB_stepsize_list)
mesh_nbtopk, mesh_dBstepsize = np.meshgrid(mesh_nbtopk, mesh_dBstepsize)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(mesh_nbtopk, mesh_dBstepsize, magnitude_feedback_proposed_codebook_topk_acc[:,:,2].T, cmap='viridis')

# Add labels and title (optional)
ax.set_xlabel('Num of narrow beam meas. feedback')
ax.set_ylabel('RSRP quan. dB stepsiee')
ax.set_zlabel('Top-3 pred. acc')
ax.set_title('3D Meshgrid Graph')

plt.show()
# Sample 2D array for the heatmap

# data = magnitude_feedback_proposed_codebook_topk_acc[:,:,2]

# # Custom axis ticks
# x_ticks = np.array(nb_topk_list)  # Custom ticks for the x-axis
# y_ticks = np.array(dB_stepsize_list)  # Custom ticks for the y-axis

# # Custom tick labels (for demonstration, using the same values as ticks)
# x_tick_labels = [str(tick) for tick in x_ticks]
# y_tick_labels = [str(tick) for tick in y_ticks]

# # Plotting the heatmap
# plt.figure(figsize=(8, 6))
# c = plt.imshow(data, cmap='hot', interpolation='nearest', aspect='auto')

# # Adjusting the ticks
# plt.xticks(x_ticks, labels=x_tick_labels)
# plt.yticks(y_ticks, labels=y_tick_labels)

# # Adding color bar
# plt.colorbar(c)

