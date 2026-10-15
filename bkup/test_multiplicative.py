# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 11:08:29 2021

@author: Yuqiang (Ethan) Heng
"""
#%%

os.chdir('/root/bkup/Learning_site_specific')
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from ComplexLayers_Torch import Beam_Classifier, Beam_Classifier_CNN, Beam_Classifier_CNN_RSRP, projection_trainer,projection_trainer_fake, projection_trainer_RSRP, fit,fit_projector, fit_projector_RSRP, eval_model
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
from beam_utils_L import ULA_DFT_codebook as DFT_codebook
from beam_utils_L import ULA_DFT_codebook_blockmatrix as DFT_codebook_blockmatrix
from beam_utils_L import plot_codebook_pattern, codebook_blockmatrix, get_AMCF_codebook, calc_beam_pattern, generate_phase_noise_matrix, generate_noise_matrix

np.random.seed(8)
# number of probing beams (N_W)
n_wide_beams = [4,8,12,16,18,20,24,32]
# number of narrow data beams to select from (N_V)
n_narrow_beams = [64 for i in n_wide_beams]
n_antenna = 64
top_k=1
nepoch = 100
batch_size = 500            
noise_factor = 0 #dB
noise_power_dBm = -81
noiseless = False

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


# h = h_real + 1j*h_imag
# valid_ue_idc = np.array([row_idx for (row_idx,row) in enumerate(np.concatenate((h_real,h_imag),axis=1)) if not all(row==0)])
# h = h[valid_ue_idc]
# h_real = h_real[valid_ue_idc]
# h_imag = h_imag[valid_ue_idc]
norm_factor = np.max(abs(h))
h_scaled = h/norm_factor
h_concat_scaled = np.concatenate((h_real/norm_factor,h_imag/norm_factor),axis=1)

distortion_trial = 10
cb_noise_list = [0]

for cb_noise_i, cb_noise in enumerate(cb_noise_list):
    if cb_noise==0:
        distortion_trial = 1
    else:
        distortion_trial = 10
        
    for distortion_iter in range (0,distortion_trial):
        print(cb_noise)
        for n_wb_i, n_wb in enumerate(n_wide_beams):
            n_nb = n_narrow_beams[n_wb_i]
            # V_noise = generate_phase_noise_matrix(n_antenna,n_nb, cb_noise)
            V_noise = generate_phase_noise_matrix(n_antenna,n_nb,cb_noise)

            # dft_nb_codebook = np.identity(n_antenna)/8

            dft_nb_codebook = DFT_codebook(nseg=n_nb,n_antenna=n_antenna)
            dft_nb_codebook_inv = np.linalg.pinv(dft_nb_codebook)
            dft_nb_codebook_tensor = torch.as_tensor(dft_nb_codebook,dtype=torch.complex64)
            dft_nb_codebook_inv_tensor = torch.as_tensor(dft_nb_codebook_inv,dtype=torch.complex64)

            # distorted_dft_nb_codebook = np.multiply(dft_nb_codebook, V_noise)
            distorted_dft_nb_codebook = np.multiply(dft_nb_codebook, V_noise)


            nb_label_clean = np.argmax(np.power(np.absolute(np.matmul(h_scaled, dft_nb_codebook.conj().T)),2),axis=1)
            nb_measurement_complex = np.matmul(h_scaled, distorted_dft_nb_codebook.conj().T)
            noise_nb_measurement_vec = np.random.normal(0,1,(nb_measurement_complex.shape[0],nb_measurement_complex.shape[1]))*np.sqrt(noise_power/2)/norm_factor + 1j*np.random.normal(0,1,(nb_measurement_complex.shape[0],nb_measurement_complex.shape[1]))*np.sqrt(noise_power/2)/norm_factor
            nb_measurement_complex = nb_measurement_complex + noise_nb_measurement_vec
            nb_measurement_abs = np.abs(nb_measurement_complex)

            
            nb_label = np.argmax(np.power(np.absolute(nb_measurement_complex),2),axis=1)
            nb_measurement_complex_concat = np.concatenate((np.real(nb_measurement_complex),np.imag(nb_measurement_complex)),axis=1)
    
            # nb_label_dirty = nb_label
            # nb_label = nb_label_clean

            train_idc, test_idc = train_test_split(np.arange(h.shape[0]),test_size=0.4)
            val_idc, test_idc = train_test_split(test_idc,test_size=0.5)



            x_train,y_train = nb_measurement_complex_concat[train_idc,:],nb_label[train_idc]
            x_val,y_val = nb_measurement_complex_concat[val_idc,:],nb_label[val_idc]
            x_test,y_test = nb_measurement_complex_concat[test_idc,:],nb_label[test_idc]

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

            x_train_fake,y_train_fake = nb_measurement_abs[train_idc,:],nb_label[train_idc]
            x_val_fake,y_val_fake = nb_measurement_abs[val_idc,:],nb_label[val_idc]
            x_test_fake,y_test_fake = nb_measurement_abs[test_idc,:],nb_label[test_idc]

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



            projection_trainer_model = projection_trainer(n_wide_beam=n_wb,n_narrow_beam=n_nb,narrow_beam_CB=dft_nb_codebook,
                                                        noise_power=noise_power,norm_factor=norm_factor)
            projection_trainer_opt = optim.Adam(projection_trainer_model.parameters(),lr=0.01, betas=(0.9,0.999), amsgrad=False)
            train_loss_hist, val_loss_hist, train_acc, val_acc = fit_projector(projection_trainer_model,top_k,dft_nb_codebook_tensor,dft_nb_codebook_inv_tensor, train_loader, val_loader, projection_trainer_opt, nn.CrossEntropyLoss(), nepoch)
            projection_trainer_model_savefname = './Saved Models/distorted_{}_projection_trainer_{}_DFT_{}_beam_probing_codebook_{}_beam_complex_input_CNN_classifier_noise_{}_dBm_tx_power_{}_dBm.pt'.format(cb_noise,dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm)
            torch.save(projection_trainer_model.state_dict(),projection_trainer_model_savefname)    
            
            projection_trainer_model_RSRP = projection_trainer_fake(n_wide_beam=n_wb,n_narrow_beam=n_nb,narrow_beam_CB=dft_nb_codebook,
                                                        noise_power=noise_power,norm_factor=norm_factor)
            projection_trainer_opt_RSRP = optim.Adam(projection_trainer_model_RSRP.parameters(),lr=0.01, betas=(0.9,0.999), amsgrad=False)
            train_loss_hist_RSRP, val_loss_hist_RSRP, train_acc_RSRP, val_acc_RSRP = fit_projector(projection_trainer_model_RSRP,top_k,dft_nb_codebook_tensor,dft_nb_codebook_inv_tensor, train_loader_fake, val_loader_fake, projection_trainer_opt_RSRP, nn.CrossEntropyLoss(), nepoch)
            projection_trainer_model_RSRP_savefname = './Saved Models/RSRP_distorted_{}_projection_trainer_{}_DFT_{}_beam_probing_codebook_{}_beam_complex_input_CNN_classifier_noise_{}_dBm_tx_power_{}_dBm.pt'.format(cb_noise,dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm)
            torch.save(projection_trainer_model_RSRP.state_dict(),projection_trainer_model_RSRP_savefname)    
            
            
            plt.figure()
            plt.plot(train_acc,label='training acc')
            plt.plot(val_acc,label='validation acc')
            plt.legend()
            plt.title('Classification acc hist with prjected narrowbeam measurements: {} wb {} nb'.format(n_wb,n_nb))
            plt.show()    
            
            projection_L_array=projection_trainer_model.get_projection_L()
            projection_L_savefname = 'distorted_{}_Top{}_projection_L_{}_{}_beam_probing_codebook_{}_beam_complex_input_CNN_classifier_noise_{}_dBm_tx_power_{}_dBm_distortion_Iter_{}.npy'.format(cb_noise,top_k,dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm,distortion_iter)
            np.save(projection_L_savefname,projection_L_array.T)

            V_noise_savefile_name = 'V_noise_{}_projection_L_{}_{}_beam_probing_codebook_{}_beam_complex_input_CNN_classifier_noise_{}_dBm_tx_power_{}_dBm_distortion_iter_{}.npy'.format(cb_noise,dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm,distortion_iter)
            np.save(V_noise_savefile_name,V_noise)
            
            projection_L_array_RSRP=projection_trainer_model_RSRP.get_projection_L()
            projection_L_savefname_RSRP = 'RSRP_distorted_{}_Top{}_projection_L_{}_{}_beam_probing_codebook_{}_beam_complex_input_CNN_classifier_noise_{}_dBm_tx_power_{}_dBm_distortion_iter_{}.npy'.format(cb_noise,top_k,dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm,distortion_iter)
            np.save(projection_L_savefname_RSRP,projection_L_array_RSRP.T)


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


for cb_noise_i, cb_noise in enumerate(cb_noise_list):
    if cb_noise==0:
        distortion_trial = 1
    else:
        distortion_trial = 10

    for distortion_iter in range(0,distortion_trial):
        print(cb_noise)
        print(distortion_iter)
        for n_wb_i, n_wb in enumerate(n_wide_beams):
            n_nb = n_narrow_beams[n_wb_i]
            print('{} Wide Beams, {} Narrow Beams.'.format(n_wb,n_nb))
            # W_noise = generate_phase_noise_matrix(n_antenna,n_wb, cb_noise).T
            W_noise = generate_phase_noise_matrix(n_antenna,n_wb, cb_noise).T


            dft_nb_codebook = DFT_codebook(nseg=n_nb,n_antenna=n_antenna)
            projection_L_savefname = 'distorted_{}_Top{}_projection_L_{}_{}_beam_probing_codebook_{}_beam_complex_input_CNN_classifier_noise_{}_dBm_tx_power_{}_dBm_distortion_Iter_{}.npy'.format(cb_noise,top_k,dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm,distortion_iter)
            unconstrained_wb_codebook = np.load(projection_L_savefname)
            unconstrained_wb_codebook = unconstrained_wb_codebook[:unconstrained_wb_codebook.shape[0]//2,:unconstrained_wb_codebook.shape[1]//2]+1j*unconstrained_wb_codebook[unconstrained_wb_codebook.shape[0]//2:,unconstrained_wb_codebook.shape[1]//2:]
            unconstrained_wb_codebook = np.matmul(unconstrained_wb_codebook,dft_nb_codebook)
            
            projection_L_RSRP_savefname = 'RSRP_distorted_{}_Top{}_projection_L_{}_{}_beam_probing_codebook_{}_beam_complex_input_CNN_classifier_noise_{}_dBm_tx_power_{}_dBm_distortion_iter_{}.npy'.format(cb_noise,top_k,dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm,distortion_iter)
            unconstrained_wb_codebook_RSRP = np.load(projection_L_RSRP_savefname)
            unconstrained_wb_codebook_RSRP = unconstrained_wb_codebook_RSRP[:unconstrained_wb_codebook_RSRP.shape[0]//2,:unconstrained_wb_codebook_RSRP.shape[1]//2]+1j*unconstrained_wb_codebook_RSRP[unconstrained_wb_codebook_RSRP.shape[0]//2:,unconstrained_wb_codebook_RSRP.shape[1]//2:]
            unconstrained_wb_codebook_RSRP = np.matmul(unconstrained_wb_codebook_RSRP,dft_nb_codebook)
            

            normalized_unconstrained_wb_codebook = unconstrained_wb_codebook * (1/np.sqrt(n_antenna)) 
            normalized_unconstrained_wb_codebook = np.multiply(normalized_unconstrained_wb_codebook,W_noise)
            
            normalized_unconstrained_wb_codebook_RSRP = unconstrained_wb_codebook_RSRP * (1/np.sqrt(n_antenna)) 
            normalized_unconstrained_wb_codebook_RSRP = np.multiply(normalized_unconstrained_wb_codebook_RSRP,W_noise)
            
            directory_path = './Saved Codebooks'
            file_name = 'distorted_{}_projection_L_{}_{}_beam_probing_codebook_{}_beam_complex_input_CNN_classifier_noise_{}_dBm_tx_power_{}_dBm_distortion_iter_{}.npy'.format(cb_noise,dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm,distortion_iter)
            full_file_path = os.path.join(directory_path, file_name)
            np.save(full_file_path, normalized_unconstrained_wb_codebook)

            file_name_RSRP = 'RSRP_distorted_{}_projection_L_{}_{}_beam_probing_codebook_{}_beam_complex_input_CNN_classifier_noise_{}_dBm_tx_power_{}_dBm_distortion_iter_{}.npy'.format(cb_noise,dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm,distortion_iter)
            full_file_path_RSRP = os.path.join(directory_path, file_name_RSRP)
            np.save(full_file_path_RSRP, normalized_unconstrained_wb_codebook_RSRP)
            

            # label = np.argmax(np.power(np.absolute(np.matmul(h_scaled, dft_nb_codebook.conj().T)),2),axis=1)
            # soft_label = np.power(np.absolute(np.matmul(h, dft_nb_codebook.conj().T)),2)
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
            
            # Proposed 
            projection_codebook_cnn_model = Beam_Classifier(n_antenna=n_antenna,n_wide_beam=n_wb,n_narrow_beam=n_nb,
                                                trainable_codebook=False,complex_codebook=normalized_unconstrained_wb_codebook,
                                                noise_power=noise_power,norm_factor=norm_factor)
            projection_codebook_cnn_opt = optim.Adam(projection_codebook_cnn_model.parameters(),lr=0.001, betas=(0.9,0.999), amsgrad=False)
            train_loss_hist, val_loss_hist, train_acc, val_acc = fit(projection_codebook_cnn_model,top_k, train_loader, val_loader, projection_codebook_cnn_opt, nn.CrossEntropyLoss(), nepoch)
            projection_cnn_model_savefname = './Saved Models/distorted_{}_Top{}_{}_DFT_{}_beam_probing_codebook_{}_beam_complex_input_CNN_classifier_noise_{}_dBm_tx_power_{}_dBm_distortion_iter_{}_RSRP.pt'.format(cb_noise,top_k,dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm,distortion_iter)
            torch.save(projection_codebook_cnn_model.state_dict(),projection_cnn_model_savefname)    
            
            plt.plot(val_acc,label='Proposed', linewidth=2)
            plt.xlabel('Epoch', fontsize=14)  
            plt.ylabel('Narrow beam pred. accuracy', fontsize=14)  
            plt.xticks(fontsize=12) 
            plt.yticks(fontsize=12) 
            plt.title('Proposed probing beam (learned w/ narrow beam measurements)')

            # # Proposed (RSRP)
            projection_codebook_cnn_model_RSRP = Beam_Classifier(n_antenna=n_antenna,n_wide_beam=n_wb,n_narrow_beam=n_nb,
                                                 trainable_codebook=False,complex_codebook=normalized_unconstrained_wb_codebook_RSRP,
                                                 noise_power=noise_power,norm_factor=norm_factor)
            projection_codebook_cnn_opt_RSRP = optim.Adam(projection_codebook_cnn_model_RSRP.parameters(),lr=0.001, betas=(0.9,0.999), amsgrad=False)
            train_loss_hist, val_loss_hist, train_acc, val_acc = fit(projection_codebook_cnn_model_RSRP,top_k, train_loader, val_loader, projection_codebook_cnn_opt_RSRP, nn.CrossEntropyLoss(), nepoch)
            projection_cnn_model_savefname_RSRP = './Saved Models/RSRP_distorted_{}_Top{}_{}_DFT_{}_beam_probing_codebook_{}_beam_complex_input_CNN_classifier_noise_{}_dBm_tx_power_{}_dBm_distortion_iter_{}_RSRP.pt'.format(cb_noise,top_k,dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm,distortion_iter)
            torch.save(projection_codebook_cnn_model_RSRP.state_dict(),projection_cnn_model_savefname_RSRP)    
            
            plt.plot(val_acc,label='Proposed (RSRP)', linewidth=2)
            plt.xlabel('Epoch', fontsize=14)  
            plt.ylabel('Narrow beam pred. accuracy', fontsize=14)  
            plt.xticks(fontsize=12) 
            plt.yticks(fontsize=12)  
            plt.title('Proposed probing beam (learned w/ narrow beam measurements)')

            
            # dft_wb_codebook = DFT_codebook(nseg=n_wb,n_antenna=n_antenna) + W_noise
            # dft_codebook_cnn_model = Beam_Classifier_CNN(n_antenna=n_antenna,n_wide_beam=n_wb,n_narrow_beam=n_nb,
            #                             trainable_codebook=False,complex_codebook=dft_wb_codebook,
            #                             noise_power=noise_power,norm_factor=norm_factor)
            # dft_codebook_cnn_opt = optim.Adam(dft_codebook_cnn_model.parameters(),lr=0.01, betas=(0.9,0.999), amsgrad=False)
            # train_loss_hist, val_loss_hist, train_acc, val_acc = fit(dft_codebook_cnn_model,top_k, train_loader, val_loader, dft_codebook_cnn_opt, nn.CrossEntropyLoss(), nepoch)
            # dft_cnn_model_savefname = './Saved Models/distorted_{}_DFTprob_Top{}_{}_DFT_{}_beam_probing_codebook_{}_beam_complex_input_CNN_classifier_noise_{}_dBm_tx_power_{}_dBm.pt'.format(cb_noise,top_k,dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm)
            # dft_codebook_savefname = 'distorted_{}_{}_DFT_{}_beam_probing_codebook_{}_beam_complex_input_CNN_classifier_noise_{}_dBm_tx_power_{}_dBm.npy'.format(cb_noise,dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm)
            # dft_codebook_save_path = os.path.join(directory_path, dft_codebook_savefname)
            # np.save(dft_codebook_save_path, dft_wb_codebook)
            # torch.save(dft_codebook_cnn_model.state_dict(),dft_cnn_model_savefname)    
            # plt.plot(val_acc,label='DFT probing beam', linewidth=2)
            
            if distortion_iter==0:
                # Trainable probing codebook + MLP classifier, MLP takes the received power as input
                learnable_codebook_model = Beam_Classifier(n_antenna=n_antenna,n_wide_beam=n_wb,n_narrow_beam=n_nb,trainable_codebook=True,noise_power=noise_power,norm_factor=norm_factor)
                learnable_codebook_opt = optim.Adam(learnable_codebook_model.parameters(),lr=0.01, betas=(0.9,0.999), amsgrad=False)
                train_loss_hist, val_loss_hist, train_acc, val_acc = fit(learnable_codebook_model,top_k, train_loader_clean, val_loader_clean, learnable_codebook_opt, nn.CrossEntropyLoss(), nepoch)  
                learnable_model_savefname = './Saved Models/Top{}_{}_trainable_{}_beam_probing_codebook_{}_beam_complex_input_CNN_classifier_noise_{}_dBm_tx_power_{}_dBm_RSRP.pt'.format(top_k,dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm)
                directory_path = './Saved Codebooks'
                
                learnable_codebook_savefname = 'Top{}_{}_trainable_{}_beam_probing_codebook_{}_beam_complex_input_CNN_classifier_noise_{}_dBm_tx_power_{}_dBm_RSRP.npy'.format(top_k,dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm)
                learnable_codebook_save_path = os.path.join(directory_path, learnable_codebook_savefname)
                learnable_codebook = learnable_codebook_model.get_codebook()
                np.save(learnable_codebook_save_path, learnable_codebook)
                torch.save(learnable_codebook_model.state_dict(),learnable_model_savefname)
           
            W_noise_savefile_name = 'W_noise_{}_projection_L_{}_{}_beam_probing_codebook_{}_beam_complex_input_CNN_classifier_noise_{}_dBm_tx_power_{}_dBm_distortion_iter_{}_RSRP.npy'.format(cb_noise,dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm,distortion_iter)
            np.save(W_noise_savefile_name,W_noise)

            
            plt.plot(val_acc,label='[10]', linewidth=2)
            plt.legend(fontsize=12)
            plt.title('{} probing beams {} narrow beams {} antennas'.format(n_wb,n_nb,n_antenna), fontsize=16)
            
            plt.show()


#%%
cb_noise=0
n_wb=16
n_nb=64
noise_power_dBm=-81
top_k=1

n_antenna=64

dataset_name = 'O1_28B_ULA'
W_noise_savefile_name = 'W_noise_{}_projection_L_{}_{}_beam_probing_codebook_{}_beam_complex_input_CNN_classifier_noise_{}_dBm_tx_power_{}_dBm_distortion_iter_0_RSRP.npy'.format(cb_noise,dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm)
W_noise_plot = np.load(W_noise_savefile_name)

for i in [1,2,3,4,5]:
    if i == 0:
        probing_codebook = np.load('./Saved Codebooks/{}_DFT_{}_beam_probing_codebook_{}_beam_complex_input_CNN_classifier_noise_{}_dBm_tx_power_{}_dBm.npy'.format(dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm))
        probing_codebook = probing_codebook.T
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111,polar=True)

        for beam_i, beam in enumerate(probing_codebook.T):
            phi, bf_gain_dft = calc_beam_pattern(beam,n_antenna=n_antenna)
            ax1.plot(phi,bf_gain_dft)
        ax1.grid(True)
        ax1.set_rmax(1)
        ax1.set_rticks([0,0.5, 1])
        ax1.set_rlabel_position(-90)  
        ax1.set_thetamin(0)
        ax1.set_thetamax(180)
        ax1.set_theta_zero_location("E")
        ax1.set_title('DFT probing beams')
        fig1.tight_layout()

    elif i == 1:
        probing_codebook = np.load('./Saved Codebooks/Top{}_{}_trainable_{}_beam_probing_codebook_{}_beam_complex_input_CNN_classifier_noise_{}_dBm_tx_power_{}_dBm_RSRP.npy'.format(top_k,dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm))
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111,polar=True)
        probing_codebook = np.multiply(probing_codebook,W_noise_plot.T)
        for beam_i, beam in enumerate(probing_codebook.T):
            phi, bf_gain_trainable = calc_beam_pattern(beam,n_antenna=n_antenna)
            ax2.plot(phi,bf_gain_trainable)
        ax2.grid(True)
        ax2.set_rmax(1)
        ax2.set_rticks([0,0.5, 1])  
        ax2.set_thetamin(0)
        ax2.set_thetamax(180)
        ax2.set_theta_zero_location("E")
        ax2.set_title('Ethans probing beams [10]')
        fig2.tight_layout()

    elif i == 2:
        probing_codebook = np.load('./Saved Codebooks/Top{}_{}_trainable_{}_beam_probing_codebook_{}_beam_complex_input_CNN_classifier_noise_{}_dBm_tx_power_{}_dBm_RSRP.npy'.format(top_k,dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm))
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111,polar=True)
        for beam_i, beam in enumerate(probing_codebook.T):
            phi, bf_gain_trainable = calc_beam_pattern(beam,n_antenna=n_antenna)
            ax2.plot(phi,bf_gain_trainable)
        ax2.grid(True)
        ax2.set_rmax(1)
        ax2.set_rticks([0,0.5, 1])  
        ax2.set_thetamin(0)
        ax2.set_thetamax(180)
        ax2.set_theta_zero_location("E")
        ax2.set_title('Ethans probing beams')
        fig2.tight_layout()

    elif i == 3:
        probing_codebook = np.load('./Saved Codebooks/distorted_{}_projection_L_{}_{}_beam_probing_codebook_{}_beam_complex_input_CNN_classifier_noise_{}_dBm_tx_power_{}_dBm_distortion_iter_0.npy'.format(cb_noise,dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm))
        probing_codebook = probing_codebook.T 
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111,polar=True)

        for beam_i, beam in enumerate(probing_codebook.T):
            phi, bf_gain_proposed = calc_beam_pattern(beam,n_antenna=n_antenna)
            ax3.plot(phi,bf_gain_proposed)
        ax3.grid(True)
        ax3.set_rmax(1)
        ax3.set_rticks([0,0.5, 1])
        ax3.set_rlabel_position(-90)  
        ax3.set_thetamin(0)
        ax3.set_thetamax(180)
        ax3.set_theta_zero_location("E")
        ax3.set_title('Proposed probing beams')
        fig3.tight_layout()

    elif i ==4:
        probing_codebook = np.load('./Saved Codebooks/distorted_{}_projection_L_{}_{}_beam_probing_codebook_{}_beam_complex_input_CNN_classifier_noise_{}_dBm_tx_power_{}_dBm_distortion_iter_0.npy'.format(cb_noise,dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm))
        probing_codebook = np.divide(probing_codebook, W_noise_plot)
        probing_codebook = probing_codebook.T 
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111,polar=True)

        for beam_i, beam in enumerate(probing_codebook.T):
            phi, bf_gain_proposed = calc_beam_pattern(beam,n_antenna=n_antenna)
            ax3.plot(phi,bf_gain_proposed)
        ax3.grid(True)
        ax3.set_rmax(1)
        ax3.set_rticks([0,0.5, 1])
        ax3.set_rlabel_position(-90)  
        ax3.set_thetamin(0)
        ax3.set_thetamax(180)
        ax3.set_theta_zero_location("E")
        ax3.set_title('Proposed probing beams with complex values')
        fig3.tight_layout()
   
    elif i == 5:
        fake_codebook = np.load('./Saved Codebooks/RSRP_distorted_{}_projection_L_{}_{}_beam_probing_codebook_{}_beam_complex_input_CNN_classifier_noise_{}_dBm_tx_power_{}_dBm_distortion_iter_{}.npy'.format(cb_noise,dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm,distortion_iter))
        # probing_codebook = np.load('./Saved Codebooks/projection_L_{}_{}_beam_probing_codebook_{}_beam_complex_input_CNN_classifier_noise_{}_dBm.npy'.format(dataset_name,n_wb,n_nb,noise_power_dBm))
        fake_codebook = fake_codebook.T
        fig4 = plt.figure()
        ax4 = fig4.add_subplot(111,polar=True)

        for beam_i, beam in enumerate(fake_codebook.T):
            phi, bf_gain_proposed = calc_beam_pattern(beam,n_antenna=n_antenna)
            ax4.plot(phi,bf_gain_proposed)
        ax4.grid(True)
        ax4.set_rmax(1)
        ax4.set_rticks([0,0.5, 1])
        ax4.set_rlabel_position(-90)  
        ax4.set_thetamin(0)
        ax4.set_thetamax(180)
        ax4.set_theta_zero_location("E")
        ax4.set_title('Proposed probing beams with abs values')
        fig4.tight_layout()
# %%
