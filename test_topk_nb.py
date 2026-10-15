# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 11:08:29 2021

@author: Yuqiang (Ethan) Heng
"""
#%%

import os
os.chdir('/root/bkup/Learning_site_specific')
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from ComplexLayers_Torch import Beam_Classifier, projection_trainer,projection_trainer_fake,  fit,fit_projector, eval_model, custom_projection_L
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
from beam_utils_L import ULA_DFT_codebook as DFT_codebook
from beam_utils_L import ULA_DFT_codebook_blockmatrix as DFT_codebook_blockmatrix
from beam_utils_L import plot_codebook_pattern, codebook_blockmatrix, get_AMCF_codebook, calc_beam_pattern, generate_phase_noise_matrix, generate_noise_matrix

np.random.seed(8)
# number of probing beams (N_W)

n_wide_beams = [16]

# number of narrow data beams to select from (N_V)
n_narrow_beams = [128 for i in n_wide_beams]
nb_topk_list=list(np.concatenate(([1,2],np.arange(4,129,4))))#1,2,4,8,16,32,64]
dB_stepsize_list=list(np.arange(0,21,2))#
n_antenna = 64
top_k=1
nepoch = 100
batch_size = 500            
noise_factor = 0 #dB
noise_power_dBm = -81
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
    h = np.load('./Dataset/downsampling_5_test_O1_28B_BS1x64x1_UE1x1x1.npy')
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

#%%

for nb_topk_ind,nb_topk in enumerate(nb_topk_list):
    print('nb_topk={}'.format(nb_topk))
    for db_stepsize_ind, dB_stepsize in enumerate(dB_stepsize_list):
        print('stepsizedB={}'.format(dB_stepsize))
        for n_wb_i, n_wb in enumerate(n_wide_beams):
            n_nb = n_narrow_beams[n_wb_i]

            dft_nb_codebook = DFT_codebook(nseg=n_nb,n_antenna=n_antenna)
            dft_nb_codebook_inv = np.linalg.pinv(dft_nb_codebook)
            dft_nb_codebook_tensor = torch.as_tensor(dft_nb_codebook,dtype=torch.complex64)
            dft_nb_codebook_inv_tensor = torch.as_tensor(dft_nb_codebook_inv,dtype=torch.complex64)


            nb_label_clean = np.argmax(np.power(np.absolute(np.matmul(h_scaled, dft_nb_codebook.conj().T)),2),axis=1)
            soft_label = np.power(np.absolute(np.matmul(h, dft_nb_codebook.conj().T)),2)

            nb_measurement_complex = np.matmul(h_scaled, dft_nb_codebook.conj().T)
            noise_nb_measurement_vec = np.random.normal(0,1,(nb_measurement_complex.shape[0],nb_measurement_complex.shape[1]))*np.sqrt(noise_power/2)/norm_factor + 1j*np.random.normal(0,1,(nb_measurement_complex.shape[0],nb_measurement_complex.shape[1]))*np.sqrt(noise_power/2)/norm_factor
            nb_measurement_complex = nb_measurement_complex + noise_nb_measurement_vec
            nb_measurement_abs = np.abs(nb_measurement_complex)

            
                    
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

            if dB_stepsize == 0:
                quantized_nb_measurement_topk_abs = nb_measurement_topk_abs
            else:    
                nb_measurement_topk_abs_dB = 10*np.log10(np.power(nb_measurement_topk_abs,2))
                quantized_nb_measurement_topk_abs_dB = np.round(nb_measurement_topk_abs_dB/dB_stepsize)*dB_stepsize
                quantized_nb_measurement_topk_abs = np.sqrt(10**(quantized_nb_measurement_topk_abs_dB/10))

            # nb_label = np.argmax(np.power(np.absolute(nb_measurement_complex),2),axis=1)
            nb_label = np.argmax(np.power(quantized_nb_measurement_topk_abs,2),axis=1)
            


            train_idc, test_idc = train_test_split(np.arange(h.shape[0]),test_size=0.4)
            val_idc, test_idc = train_test_split(test_idc,test_size=0.5)


            x_train,y_train = nb_measurement_complex_topk_concat[train_idc,:],nb_label[train_idc]
            x_val,y_val = nb_measurement_complex_topk_concat[val_idc,:],nb_label[val_idc]
            x_test,y_test = nb_measurement_complex_topk_concat[test_idc,:],nb_label[test_idc]

            torch_x_train = torch.from_numpy(x_train)
            torch_y_train = torch.from_numpy(y_train)
            torch_x_val = torch.from_numpy(x_val)
            torch_y_val = torch.from_numpy(y_val)
            torch_x_test = torch.from_numpy(x_test)
            torch_y_test = torch.from_numpy(y_test)

            # Pytorch train and test sets
            train = torch.utils.data.TensorDataset(torch_x_train,torch_y_train)
            val = torch.utils.data.TensorDataset(torch_x_val,torch_y_val)
            test = torch.utils.data.TensorDataset(torch_x_test,torch_y_test)

            # data loader
            train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)
            val_loader = torch.utils.data.DataLoader(val, batch_size = batch_size, shuffle = False)
            test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)    

            x_train_fake,y_train_fake = quantized_nb_measurement_topk_abs[train_idc,:],nb_label[train_idc]
            x_val_fake,y_val_fake = quantized_nb_measurement_topk_abs[val_idc,:],nb_label[val_idc]
            x_test_fake,y_test_fake = quantized_nb_measurement_topk_abs[test_idc,:],nb_label[test_idc]

            torch_x_train_fake = torch.from_numpy(x_train_fake)
            torch_y_train_fake = torch.from_numpy(y_train_fake)
            torch_x_val_fake = torch.from_numpy(x_val_fake)
            torch_y_val_fake = torch.from_numpy(y_val_fake)
            torch_x_test_fake = torch.from_numpy(x_test_fake)
            torch_y_test_fake = torch.from_numpy(y_test_fake)

            # Pytorch train and test sets

            train_fake = torch.utils.data.TensorDataset(torch_x_train_fake,torch_y_train_fake)
            val_fake = torch.utils.data.TensorDataset(torch_x_val_fake,torch_y_val_fake)
            test_fake = torch.utils.data.TensorDataset(torch_x_test_fake,torch_y_test_fake)


            # data loader
            train_loader_fake = torch.utils.data.DataLoader(train_fake, batch_size = batch_size, shuffle = False)
            val_loader_fake = torch.utils.data.DataLoader(val_fake, batch_size = batch_size, shuffle = False)
            test_loader_fake = torch.utils.data.DataLoader(test_fake, batch_size = batch_size, shuffle = False)    



            # complex_feedback_projection_trainer_model = projection_trainer(n_wide_beam=n_wb,n_narrow_beam=n_nb,narrow_beam_CB=dft_nb_codebook,
            #                                             noise_power=noise_power,norm_factor=norm_factor)
            # complex_feedback_projection_trainer_opt = optim.Adam(complex_feedback_projection_trainer_model.parameters(),lr=0.01, betas=(0.9,0.999), amsgrad=False)
            # train_loss_hist, val_loss_hist, train_acc, val_acc = fit_projector(complex_feedback_projection_trainer_model,top_k,dft_nb_codebook_tensor,dft_nb_codebook_inv_tensor, train_loader, val_loader, complex_feedback_projection_trainer_opt, nn.CrossEntropyLoss(), nepoch)
            # complex_feedback_projection_trainer_model_savefname = './Saved Projection Models/lowcomplexity_top{}_{}dBstepsize_complex_feedback_projection_trainer_{}_{}_beam_probing_codebook_{}_beam_noise_{}_dBm_tx_power_{}_dBm.pt'.format(nb_topk,dB_stepsize,dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm)
            # torch.save(complex_feedback_projection_trainer_model.state_dict(),complex_feedback_projection_trainer_model_savefname)    

            
            if_fake = True
            magnitude_feedback_projection_trainer_model = projection_trainer_fake(if_fake,n_wide_beam=n_wb,n_narrow_beam=n_nb,narrow_beam_CB=dft_nb_codebook,
                                                        noise_power=noise_power,norm_factor=norm_factor)
            magnitude_feedback_projection_trainer_opt = optim.Adam(magnitude_feedback_projection_trainer_model.parameters(),lr=0.01, betas=(0.9,0.999), amsgrad=False)
            train_loss_hist_RSRP, val_loss_hist_RSRP, train_acc_RSRP, val_acc_RSRP = fit_projector(magnitude_feedback_projection_trainer_model,top_k,dft_nb_codebook_tensor,dft_nb_codebook_inv_tensor, train_loader_fake, val_loader_fake, magnitude_feedback_projection_trainer_opt, nn.CrossEntropyLoss(), nepoch)
            magnitude_feedback_projection_trainer_model_savefname = './Saved Projection Models/lowcomplexity_top{}_{}dBstepsize_magnitude_feedback_projection_trainer_{}_{}_beam_probing_codebook_{}_beam_noise_{}_dBm_tx_power_{}_dBm.pt'.format(nb_topk,dB_stepsize,dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm)
            torch.save(magnitude_feedback_projection_trainer_model.state_dict(),magnitude_feedback_projection_trainer_model_savefname)    
            
            
            # plt.figure()
            # plt.plot(train_acc,label='training acc')
            # plt.plot(val_acc,label='validation acc')
            # plt.legend()
            # plt.title('Classification acc hist with prjected narrowbeam measurements: {} wb {} nb'.format(n_wb,n_nb))
            # plt.show()    
            
            # complex_feedback_projection_matrix = complex_feedback_projection_trainer_model.get_projection_L()
            # complex_feedback_projection_matrix_savefname = './Saved Projections/lowcomplexity_top{}_{}dBstepsize_complex_feedback_projection_matrix_{}_{}_beam_probing_codebook_{}_beam_noise_{}_dBm_tx_power_{}_dBm.npy'.format(nb_topk,dB_stepsize,dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm)
            # np.save(complex_feedback_projection_matrix_savefname,complex_feedback_projection_matrix.T)
            
            
            magnitude_feedback_projection_matrix = magnitude_feedback_projection_trainer_model.get_projection_L()
            magnitude_feedback_projection_matrix_savefname = './Saved Projections/lowcomplexity_top{}_{}dBstepsize_magnitude_feedback_projection_matrix_{}_{}_beam_probing_codebook_{}_beam_noise_{}_dBm_tx_power_{}_dBm.npy'.format(nb_topk,dB_stepsize,dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm)
            np.save(magnitude_feedback_projection_matrix_savefname,magnitude_feedback_projection_matrix.T)


top_k=1

# Compute EGC gain
egc_gain_scaled = np.power(np.sum(abs(h_scaled),axis=1),2)/n_antenna
train_idc, test_idc = train_test_split(np.arange(h.shape[0]),test_size=0.4)
val_idc, test_idc = train_test_split(test_idc,test_size=0.5)


x_train,y_train = h_concat_scaled[train_idc,:],egc_gain_scaled[train_idc]
x_val,y_val = h_concat_scaled[val_idc,:],egc_gain_scaled[val_idc]
x_test,y_test = h_concat_scaled[test_idc,:],egc_gain_scaled[test_idc]

torch_x_train = torch.from_numpy(x_train)
torch_y_train = torch.from_numpy(y_train)
torch_x_val = torch.from_numpy(x_val)
torch_y_val = torch.from_numpy(y_val)
torch_x_test = torch.from_numpy(x_test)
torch_y_test = torch.from_numpy(y_test)

# Pytorch train and test sets



train = torch.utils.data.TensorDataset(torch_x_train,torch_y_train)
val = torch.utils.data.TensorDataset(torch_x_val,torch_y_val)
test = torch.utils.data.TensorDataset(torch_x_test,torch_y_test)

# data loader
train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)
val_loader = torch.utils.data.DataLoader(val, batch_size = batch_size, shuffle = False)
test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)    
plt.figure()   


for nb_topk_ind,nb_topk in enumerate(nb_topk_list):
    for db_stepsize_ind, dB_stepsize in enumerate(dB_stepsize_list):

        for n_wb_i, n_wb in enumerate(n_wide_beams):
            n_nb = n_narrow_beams[n_wb_i]
            print('{} Wide Beams, {} Narrow Beams.'.format(n_wb,n_nb))

            dft_nb_codebook = DFT_codebook(nseg=n_nb,n_antenna=n_antenna)
            # complex_feedback_projection_matrix_savefname = './Saved Projections/lowcomplexity_top{}_{}dBstepsize_complex_feedback_projection_matrix_{}_{}_beam_probing_codebook_{}_beam_noise_{}_dBm_tx_power_{}_dBm.npy'.format(nb_topk,dB_stepsize,dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm)
            # complex_feedback_projection_matrix_blockmatrix = np.load(complex_feedback_projection_matrix_savefname)
            # complex_feedback_projection_matrix = complex_feedback_projection_matrix_blockmatrix[:complex_feedback_projection_matrix_blockmatrix.shape[0]//2,:complex_feedback_projection_matrix_blockmatrix.shape[1]//2]+1j*complex_feedback_projection_matrix_blockmatrix[complex_feedback_projection_matrix_blockmatrix.shape[0]//2:,:complex_feedback_projection_matrix_blockmatrix.shape[1]//2]
            # complex_feedback_proposed_wb_codebook = np.matmul(complex_feedback_projection_matrix.conj(),dft_nb_codebook)#* (1/np.sqrt(n_antenna))
            
            magnitude_feedback_projection_matrix_savefname = './Saved Projections/lowcomplexity_top{}_{}dBstepsize_magnitude_feedback_projection_matrix_{}_{}_beam_probing_codebook_{}_beam_noise_{}_dBm_tx_power_{}_dBm.npy'.format(nb_topk,dB_stepsize,dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm)
            magnitude_feedback_projection_matrix_blockmatrix = np.load(magnitude_feedback_projection_matrix_savefname)
            magnitude_feedback_projection_matrix = magnitude_feedback_projection_matrix_blockmatrix[:magnitude_feedback_projection_matrix_blockmatrix.shape[0]//2,:magnitude_feedback_projection_matrix_blockmatrix.shape[1]//2]+1j*magnitude_feedback_projection_matrix_blockmatrix[magnitude_feedback_projection_matrix_blockmatrix.shape[0]//2:,:magnitude_feedback_projection_matrix_blockmatrix.shape[1]//2]
            magnitude_feedback_proposed_wb_codebook = np.matmul(magnitude_feedback_projection_matrix.conj(),dft_nb_codebook)#* (1/np.sqrt(n_antenna))
            

            
            directory_path = './Saved Codebooks'
            # file_name = 'lowcomplexity_top{}_{}dBstepsize_complex_feedback_proposed_codebook_{}_{}_beam_probing_codebook_{}_beam_noise_{}_dBm_tx_power_{}_dBm.npy'.format(nb_topk,dB_stepsize,dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm)
            # full_file_path = os.path.join(directory_path, file_name)
            # np.save(full_file_path, complex_feedback_proposed_wb_codebook)

            file_name_RSRP = 'lowcomplexity_top{}_{}dBstepsize_magnitude_feedback_proposed_codebook_{}_{}_beam_probing_codebook_{}_beam_noise_{}_dBm_tx_power_{}_dBm.npy'.format(nb_topk,dB_stepsize,dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm)
            full_file_path_RSRP = os.path.join(directory_path, file_name_RSRP)
            np.save(full_file_path_RSRP, magnitude_feedback_proposed_wb_codebook)

            label = nb_label
            label_clean = nb_label_clean

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
            
            
            # clean nb label, for ethan 
            x_train_clean,y_train_clean = h_concat_scaled[train_idc,:],label_clean[train_idc]
            x_val_clean,y_val_clean = h_concat_scaled[val_idc,:],label_clean[val_idc]
            x_test_clean,y_test_clean = h_concat_scaled[test_idc,:],label_clean[test_idc]
            
            torch_x_train_clean,torch_y_train_clean = torch.from_numpy(x_train_clean),torch.from_numpy(y_train_clean)
            torch_x_val_clean,torch_y_val_clean = torch.from_numpy(x_val_clean),torch.from_numpy(y_val_clean)
            torch_x_test_clean,torch_y_test_clean = torch.from_numpy(x_test_clean),torch.from_numpy(y_test_clean)
            
            # Pytorch train and test sets
            train_clean = torch.utils.data.TensorDataset(torch_x_train_clean,torch_y_train_clean)
            val_clean = torch.utils.data.TensorDataset(torch_x_val_clean,torch_y_val_clean)
            test_clean = torch.utils.data.TensorDataset(torch_x_test_clean,torch_y_test_clean)
            
            # data loader
            train_loader_clean = torch.utils.data.DataLoader(train_clean, batch_size = batch_size, shuffle = False)
            val_loader_clean = torch.utils.data.DataLoader(val_clean, batch_size = batch_size, shuffle = False)
            test_loader_clean = torch.utils.data.DataLoader(test_clean, batch_size = batch_size, shuffle = False)
            
            # Proposed (complex)
            # complex_feedback_propsed_codebook_model = Beam_Classifier(n_antenna=n_antenna,n_wide_beam=n_wb,n_narrow_beam=n_nb,
            #                                     trainable_codebook=False,complex_codebook=complex_feedback_proposed_wb_codebook,
            #                                     noise_power=noise_power,norm_factor=norm_factor)
            # complex_feedback_propsed_codebook_opt = optim.Adam(complex_feedback_propsed_codebook_model.parameters(),lr=0.001, betas=(0.9,0.999), amsgrad=False)
            # train_loss_hist, val_loss_hist, train_acc, val_acc = fit(complex_feedback_propsed_codebook_model,top_k, train_loader, val_loader, complex_feedback_propsed_codebook_opt, nn.CrossEntropyLoss(), nepoch)
            # complex_feedback_propsed_codebook_model_savefname = './Saved Models/lowcomplexity_top{}_{}dBstepsize_complex_feedback_propsed_{}_{}_beam_probing_codebook_{}_beam_noise_{}_dBm_tx_power_{}_dBm.pt'.format(nb_topk,dB_stepsize,dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm)
            # torch.save(complex_feedback_propsed_codebook_model.state_dict(),complex_feedback_propsed_codebook_model_savefname)    
            
            # plt.plot(val_acc,label='Proposed (complex)', linewidth=2)
            # plt.xlabel('Epoch', fontsize=14)  
            # plt.ylabel('Narrow beam pred. accuracy', fontsize=14)  

            # plt.xticks(fontsize=12) 
            # plt.yticks(fontsize=12) 
            # plt.title('Proposed probing beam (learned w/ narrow beam measurements)')

            # # Proposed (magnitude)
            magnitude_feedback_proposed_codebook_model = Beam_Classifier(n_antenna=n_antenna,n_wide_beam=n_wb,n_narrow_beam=n_nb,
                                                    trainable_codebook=False,complex_codebook=magnitude_feedback_proposed_wb_codebook,
                                                    noise_power=noise_power,norm_factor=norm_factor)
            magnitude_feedback_proposed_codebook_opt = optim.Adam(magnitude_feedback_proposed_codebook_model.parameters(),lr=0.001, betas=(0.9,0.999), amsgrad=False)
            train_loss_hist, val_loss_hist, train_acc, val_acc = fit(magnitude_feedback_proposed_codebook_model,top_k, train_loader, val_loader, magnitude_feedback_proposed_codebook_opt, nn.CrossEntropyLoss(), nepoch)
            magnitude_feedback_proposed_codebook_model_savefname = './Saved Models/lowcomplexity_top{}_{}dBstepsize_magnitude_feedback_propsed_{}_{}_beam_probing_codebook_{}_beam_noise_{}_dBm_tx_power_{}_dBm.pt'.format(nb_topk,dB_stepsize,dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm)
            torch.save(magnitude_feedback_proposed_codebook_model.state_dict(),magnitude_feedback_proposed_codebook_model_savefname)    
            
            plt.plot(val_acc,label='Proposed (magnitude)', linewidth=2)
            plt.xlabel('Epoch', fontsize=14)  
            plt.ylabel('Narrow beam pred. accuracy', fontsize=14)  
            plt.xticks(fontsize=12) 
            plt.yticks(fontsize=12)  
            plt.title('Proposed probing beam (learned w/ narrow beam measurements)')

            
            # dft_wb_codebook = DFT_codebook(nseg=n_wb,n_antenna=n_antenna)
            # dft_codebook_cnn_model = Beam_Classifier(n_antenna=n_antenna,n_wide_beam=n_wb,n_narrow_beam=n_nb,
            #                             trainable_codebook=False,complex_codebook=dft_wb_codebook,
            #                             noise_power=noise_power,norm_factor=norm_factor)
            # dft_codebook_cnn_opt = optim.Adam(dft_codebook_cnn_model.parameters(),lr=0.01, betas=(0.9,0.999), amsgrad=False)
            # train_loss_hist, val_loss_hist, train_acc, val_acc = fit(dft_codebook_cnn_model,top_k, train_loader, val_loader, dft_codebook_cnn_opt, nn.CrossEntropyLoss(), nepoch)
            # dft_cnn_model_savefname = './Saved Models/DFTprob_{}_{}_beam_probing_codebook_{}_beam_noise_{}_dBm_tx_power_{}_dBm.pt'.format(dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm)
            # torch.save(dft_codebook_cnn_model.state_dict(),dft_cnn_model_savefname)    
            # plt.plot(val_acc,label='DFT probing beam', linewidth=2)
            
            # # Trainable probing codebook + MLP classifier, MLP takes the received power as input
            # learnable_codebook_model = Beam_Classifier(n_antenna=n_antenna,n_wide_beam=n_wb,n_narrow_beam=n_nb,trainable_codebook=True,noise_power=noise_power,norm_factor=norm_factor)
            # learnable_codebook_opt = optim.Adam(learnable_codebook_model.parameters(),lr=0.01, betas=(0.9,0.999), amsgrad=False)
            # train_loss_hist, val_loss_hist, train_acc, val_acc = fit(learnable_codebook_model,top_k, train_loader_clean, val_loader_clean, learnable_codebook_opt, nn.CrossEntropyLoss(), nepoch)  
            # learnable_model_savefname = './Saved Models/Ethan_{}_{}_beam_probing_codebook_{}_beam_noise_{}_dBm_tx_power_{}_dBm.pt'.format(dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm)
            # directory_path = './Saved Codebooks'
            
            # learnable_codebook_savefname = 'Ethan_{}_{}_beam_probing_codebook_{}_beam_noise_{}_dBm_tx_power_{}_dBm.npy'.format(dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm)
            # learnable_codebook_save_path = os.path.join(directory_path, learnable_codebook_savefname)
            # learnable_codebook = learnable_codebook_model.get_codebook()
            # np.save(learnable_codebook_save_path, learnable_codebook)
            # torch.save(learnable_codebook_model.state_dict(),learnable_model_savefname)

                        
            # plt.plot(val_acc,label='[10]', linewidth=2)
            # plt.legend(fontsize=12)
            # plt.title('{} probing beams {} narrow beams {} antennas'.format(n_wb,n_nb,n_antenna), fontsize=16)
            
            plt.show()


#%%
n_wb=16
n_nb=64
noise_power_dBm=-81
top_k=1

n_antenna=64

dataset_name = 'O1_28B_ULA'

for i in [0,1,2,3]:
    if i == 0:
        dft_probing_codebook =  DFT_codebook(nseg=n_wb,n_antenna=n_antenna).T
        fig0 = plt.figure()
        ax0 = fig0.add_subplot(111,polar=True)

        for beam_i, beam in enumerate(dft_probing_codebook.T):
            phi, bf_gain_dft = calc_beam_pattern(beam,n_antenna=n_antenna)
            ax0.plot(phi,bf_gain_dft)
        ax0.grid(True)
        ax0.set_rticks([0,0.5, 1])
        ax0.set_rlabel_position(-90)  
        ax0.set_thetamin(0)
        ax0.set_thetamax(180)
        ax0.set_theta_zero_location("E")
        ax0.set_title('DFT probing beams')
        fig0.tight_layout()

    elif i == 1:
        Ethan_codebook = np.load('./Saved Codebooks/Ethan_{}_{}_beam_probing_codebook_{}_beam_noise_{}_dBm_tx_power_{}_dBm.npy'.format(dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm))
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111,polar=True)
        for beam_i, beam in enumerate(Ethan_codebook.T):
            phi, bf_gain_trainable = calc_beam_pattern(beam,n_antenna=n_antenna)
            ax1.plot(phi,bf_gain_trainable)
        ax1.grid(True)
        ax1.set_rmax(1)
        ax1.set_rticks([0,0.5, 1])  
        ax1.set_thetamin(0)
        ax1.set_thetamax(180)
        ax1.set_theta_zero_location("E")
        ax1.set_title('Learned w/ explicit channel knowledge [12]')
        fig1.tight_layout()


    elif i == 2:
        complex_feed_proposed_probing_codebook = np.load('./Saved Codebooks/lowcomplexity_top{}_complex_feedback_proposed_codebook_{}_{}_beam_probing_codebook_{}_beam_noise_{}_dBm_tx_power_{}_dBm.npy'.format(nb_topk,dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm))
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111,polar=True)

        for beam_i, beam in enumerate(complex_feed_proposed_probing_codebook):
            phi, bf_gain_proposed = calc_beam_pattern(beam,n_antenna=n_antenna)
            ax2.plot(phi,bf_gain_proposed)
        ax2.grid(True)
        ax2.set_rmax(1)
        ax2.set_rticks([0,0.5, 1])
        ax2.set_rlabel_position(-90)  
        ax2.set_thetamin(0)
        ax2.set_thetamax(180)
        ax2.set_theta_zero_location("E")
        ax2.set_title('Proposed probing beams with complex value feeback')
        fig2.tight_layout()

    elif i == 3:
        magnitude_feed_proposed_probing_codebook = np.load('./Saved Codebooks/lowcomplexity_top{}_magnitude_feedback_proposed_codebook_{}_{}_beam_probing_codebook_{}_beam_noise_{}_dBm_tx_power_{}_dBm.npy'.format(nb_topk,dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm))
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111,polar=True)

        for beam_i, beam in enumerate(magnitude_feed_proposed_probing_codebook):
            phi, bf_gain_proposed = calc_beam_pattern(beam,n_antenna=n_antenna)
            ax3.plot(phi,bf_gain_proposed)
        ax3.grid(True)

        ax3.set_rmax(1)
        ax3.set_rticks([0,0.5, 1])
        ax3.set_rlabel_position(-90)  
        ax3.set_thetamin(0)
        ax3.set_thetamax(180)
        ax3.set_theta_zero_location("E")
        ax3.set_title('Proposed probing beams with magnitude value feedback')
        fig3.tight_layout()

#%%
