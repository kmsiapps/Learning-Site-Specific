# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 11:08:29 2021

@author: Yuqiang (Ethan) Heng
"""
#%%

# os.chdir('/root/bkup/Learning_site_specific')
import os
import argparse
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from ComplexLayers_Torch import Beam_Classifier, projection_trainer,projection_trainer_fake,  fit,fit_projector,  eval_model, custom_projection_L
from train_norandomphase import fit_projector_device
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
from beam_utils_L import ULA_DFT_codebook as DFT_codebook
from beam_utils_L import ULA_DFT_codebook_blockmatrix as DFT_codebook_blockmatrix
from beam_utils_L import plot_codebook_pattern, codebook_blockmatrix, get_AMCF_codebook, calc_beam_pattern, generate_phase_noise_matrix, generate_noise_matrix
np.random.seed(8)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_wide_beams", type=int, nargs="+", default=[16])
    parser.add_argument("--noise_power_dBm_list", type=float, nargs="+", default=[-81])
    parser.add_argument("--dataset_name", type=str, default="O1_28B_ULA", choices=["O1_28B_ULA", "Rosslyn_ULA", "I3_60_ULA", "O1_28_ULA"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--train_only", action="store_true", default=True)
    return parser.parse_args()

args = parse_args()
device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
print('=' * 50)
print(f"Probing beams (N_W): {args.n_wide_beams}")
print(f"Noise power (dBm): {args.noise_power_dBm_list}")
print(f"Dataset name: {args.dataset_name}")
print(f"Device: {device}")

# number of probing beams (N_W)

n_wide_beams = args.n_wide_beams
# number of narrow data beams to select from (N_V)
n_narrow_beams = [64 for i in n_wide_beams]

n_antenna = 64
top_k=1
nepoch = args.epochs
batch_size = args.batch_size            
noise_factor = 0 #dB
noise_power_dBm_list = args.noise_power_dBm_list 
noiseless = False

dataset_name = args.dataset_name # 'Rosslyn_ULA' or 'O1_28B_ULA' or 'I3_60_ULA' or 'O1_28_ULA'
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


norm_factor = np.max(abs(h))
h_scaled = h/norm_factor
h_concat_scaled = np.concatenate((h_real/norm_factor,h_imag/norm_factor),axis=1)
for noise_power_dBm_ind, noise_power_dBm in enumerate(noise_power_dBm_list):
    noise_power = 10**((noise_power_dBm-tx_power_dBm-noise_factor)/10)
    for n_wb_i, n_wb in enumerate(n_wide_beams):
        n_nb = n_narrow_beams[n_wb_i]
        
        dft_nb_codebook = DFT_codebook(nseg=n_nb,n_antenna=n_antenna)
        dft_nb_codebook_inv = np.linalg.pinv(dft_nb_codebook)
        dft_nb_codebook_tensor = torch.as_tensor(dft_nb_codebook,dtype=torch.complex64, device=device)
        dft_nb_codebook_inv_tensor = torch.as_tensor(dft_nb_codebook_inv,dtype=torch.complex64, device=device)


        nb_label_clean = np.argmax(np.power(np.absolute(np.matmul(h_scaled, dft_nb_codebook.conj().T)),2),axis=1)
        soft_label = np.power(np.absolute(np.matmul(h, dft_nb_codebook.conj().T)),2)

        nb_measurement_complex = np.matmul(h_scaled, dft_nb_codebook.conj().T)
        noise_nb_measurement_vec = np.random.normal(0,1,(nb_measurement_complex.shape[0],nb_measurement_complex.shape[1]))*np.sqrt(noise_power/2)/norm_factor + 1j*np.random.normal(0,1,(nb_measurement_complex.shape[0],nb_measurement_complex.shape[1]))*np.sqrt(noise_power/2)/norm_factor
        nb_measurement_complex = nb_measurement_complex + noise_nb_measurement_vec
        nb_measurement_abs = np.abs(nb_measurement_complex)

        
        nb_label = np.argmax(np.power(np.absolute(nb_measurement_complex),2),axis=1)
        
        nb_measurement_complex_concat = np.concatenate((np.real(nb_measurement_complex),np.imag(nb_measurement_complex)),axis=1)

    

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
        pin_memory = device.type == "cuda"
        loader_kwargs = dict(batch_size=batch_size, shuffle=False, pin_memory=pin_memory,
                             num_workers=args.num_workers, persistent_workers=args.num_workers > 0)
        train_loader_fake = torch.utils.data.DataLoader(train_fake, **loader_kwargs)
        val_loader_fake = torch.utils.data.DataLoader(val_fake, **loader_kwargs)
        test_loader_fake = torch.utils.data.DataLoader(test_fake, **loader_kwargs)    



        # complex_feedback_projection_trainer_model = projection_trainer(n_wide_beam=n_wb,n_narrow_beam=n_nb,narrow_beam_CB=dft_nb_codebook,
        #                                             noise_power=noise_power,norm_factor=norm_factor)
        # complex_feedback_projection_trainer_opt = optim.Adam(complex_feedback_projection_trainer_model.parameters(),lr=0.01, betas=(0.9,0.999), amsgrad=False)
        # train_loss_hist, val_loss_hist, train_acc, val_acc = fit_projector(complex_feedback_projection_trainer_model,top_k,dft_nb_codebook_tensor,dft_nb_codebook_inv_tensor, train_loader, val_loader, complex_feedback_projection_trainer_opt, nn.CrossEntropyLoss(), nepoch)
        # complex_feedback_projection_trainer_model_savefname = './Saved Projection Models/complex_feedback_projection_trainer_{}_{}_beam_probing_codebook_{}_beam_noise_{}_dBm_tx_power_{}_dBm.pt'.format(dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm)
        # torch.save(complex_feedback_projection_trainer_model.state_dict(),complex_feedback_projection_trainer_model_savefname)    

        
        magnitude_feedback_projection_trainer_model_savefname = './Saved Projection Models/magnitude_feedback_projection_trainer_{}_{}_beam_probing_codebook_{}_beam_noise_{}_dBm_tx_power_{}_dBm.pt'.format(dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm)
        magnitude_feedback_projection_matrix_savefname = './Saved Projections/magnitude_feedback_projection_matrix_{}_{}_beam_probing_codebook_{}_beam_noise_{}_dBm_tx_power_{}_dBm.npy'.format(dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm)
        if os.path.exists(magnitude_feedback_projection_trainer_model_savefname) and os.path.exists(magnitude_feedback_projection_matrix_savefname) and not args.overwrite:
            print(f'[skip] projection artifacts exist for N_W={n_wb}, noise={noise_power_dBm}. Use --overwrite to retrain.')
            continue

        if_fake = True
        magnitude_feedback_projection_trainer_model = projection_trainer_fake(if_fake,n_wide_beam=n_wb,n_narrow_beam=n_nb,narrow_beam_CB=dft_nb_codebook,
                                                    noise_power=noise_power,norm_factor=norm_factor).to(device)
        magnitude_feedback_projection_trainer_opt = optim.Adam(magnitude_feedback_projection_trainer_model.parameters(),lr=0.01, betas=(0.9,0.999), amsgrad=False)
        train_loss_hist_RSRP, val_loss_hist_RSRP, train_acc_RSRP, val_acc_RSRP = fit_projector_device(magnitude_feedback_projection_trainer_model,top_k,dft_nb_codebook_tensor,dft_nb_codebook_inv_tensor, train_loader_fake, val_loader_fake, magnitude_feedback_projection_trainer_opt, nn.CrossEntropyLoss(), nepoch, device)
        torch.save(magnitude_feedback_projection_trainer_model.cpu().state_dict(),magnitude_feedback_projection_trainer_model_savefname)    
        
        
        plt.figure()
        plt.plot(train_acc_RSRP,label='training acc')
        plt.plot(val_acc_RSRP,label='validation acc')
        plt.legend()
        plt.title('Classification acc hist with prjected narrowbeam measurements: {} wb {} nb'.format(n_wb,n_nb))
        if not os.environ.get("MPLBACKEND") == "Agg":
            plt.show()    
        
        # complex_feedback_projection_matrix = complex_feedback_projection_trainer_model.get_projection_L()
        # complex_feedback_projection_matrix_savefname = './Saved Projections/complex_feedback_projection_matrix_{}_{}_beam_probing_codebook_{}_beam_noise_{}_dBm_tx_power_{}_dBm.npy'.format(dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm)
        # np.save(complex_feedback_projection_matrix_savefname,complex_feedback_projection_matrix.T)
        
        
        magnitude_feedback_projection_matrix = magnitude_feedback_projection_trainer_model.get_projection_L()
        np.save(magnitude_feedback_projection_matrix_savefname,magnitude_feedback_projection_matrix.T)






top_k=1
# Compute EGC gain
egc_gain_scaled = np.power(np.sum(abs(h_scaled),axis=1),2)/n_antenna
# train_idc, test_idc = train_test_split(np.arange(h.shape[0]),test_size=0.4)
# val_idc, test_idc = train_test_split(test_idc,test_size=0.5)


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


for noise_power_dBm_ind, noise_power_dBm in enumerate(noise_power_dBm_list):
    noise_power = 10**((noise_power_dBm-tx_power_dBm-noise_factor)/10)
    for n_wb_i, n_wb in enumerate(n_wide_beams):
        n_nb = n_narrow_beams[n_wb_i]
        print('{} Wide Beams, {} Narrow Beams.'.format(n_wb,n_nb))

        dft_nb_codebook = DFT_codebook(nseg=n_nb,n_antenna=n_antenna)
        # complex_feedback_projection_matrix_savefname = './Saved Projections/complex_feedback_projection_matrix_{}_{}_beam_probing_codebook_{}_beam_noise_{}_dBm_tx_power_{}_dBm.npy'.format(dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm)
        # complex_feedback_projection_matrix_blockmatrix = np.load(complex_feedback_projection_matrix_savefname)
        # complex_feedback_projection_matrix = complex_feedback_projection_matrix_blockmatrix[:complex_feedback_projection_matrix_blockmatrix.shape[0]//2,:complex_feedback_projection_matrix_blockmatrix.shape[1]//2]+1j*complex_feedback_projection_matrix_blockmatrix[complex_feedback_projection_matrix_blockmatrix.shape[0]//2:,:complex_feedback_projection_matrix_blockmatrix.shape[1]//2]
        # complex_feedback_proposed_wb_codebook = np.matmul(complex_feedback_projection_matrix.conj(),dft_nb_codebook)#* (1/np.sqrt(n_antenna))
        
        magnitude_feedback_projection_matrix_savefname = './Saved Projections/magnitude_feedback_projection_matrix_{}_{}_beam_probing_codebook_{}_beam_noise_{}_dBm_tx_power_{}_dBm.npy'.format(dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm)
        magnitude_feedback_projection_matrix_blockmatrix = np.load(magnitude_feedback_projection_matrix_savefname)
        magnitude_feedback_projection_matrix = magnitude_feedback_projection_matrix_blockmatrix[:magnitude_feedback_projection_matrix_blockmatrix.shape[0]//2,:magnitude_feedback_projection_matrix_blockmatrix.shape[1]//2]+1j*magnitude_feedback_projection_matrix_blockmatrix[magnitude_feedback_projection_matrix_blockmatrix.shape[0]//2:,:magnitude_feedback_projection_matrix_blockmatrix.shape[1]//2]
        magnitude_feedback_proposed_wb_codebook = np.matmul(magnitude_feedback_projection_matrix.conj(),dft_nb_codebook)#* (1/np.sqrt(n_antenna))
        

        
        directory_path = './Saved Codebooks'
        # file_name = 'complex_feedback_proposed_codebook_{}_{}_beam_probing_codebook_{}_beam_noise_{}_dBm_tx_power_{}_dBm.npy'.format(dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm)
        # full_file_path = os.path.join(directory_path, file_name)
        # np.save(full_file_path, complex_feedback_proposed_wb_codebook)

        file_name_RSRP = 'magnitude_feedback_proposed_codebook_{}_{}_beam_probing_codebook_{}_beam_noise_{}_dBm_tx_power_{}_dBm.npy'.format(dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm)
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
        pin_memory = device.type == "cuda"
        loader_kwargs = dict(batch_size=batch_size, shuffle=False, pin_memory=pin_memory,
                             num_workers=args.num_workers, persistent_workers=args.num_workers > 0)
        train_loader_clean = torch.utils.data.DataLoader(train_clean, **loader_kwargs)
        val_loader_clean = torch.utils.data.DataLoader(val_clean, **loader_kwargs)
        test_loader_clean = torch.utils.data.DataLoader(test_clean, **loader_kwargs)
        
        # Proposed (complex)
        # complex_feedback_propsed_codebook_model = Beam_Classifier(n_antenna=n_antenna,n_wide_beam=n_wb,n_narrow_beam=n_nb,
        #                                     trainable_codebook=False,complex_codebook=complex_feedback_proposed_wb_codebook,
        #                                     noise_power=noise_power,norm_factor=norm_factor)
        # complex_feedback_propsed_codebook_opt = optim.Adam(complex_feedback_propsed_codebook_model.parameters(),lr=0.001, betas=(0.9,0.999), amsgrad=False)
        # train_loss_hist, val_loss_hist, train_acc, val_acc = fit(complex_feedback_propsed_codebook_model,top_k, train_loader, val_loader, complex_feedback_propsed_codebook_opt, nn.CrossEntropyLoss(), nepoch)
        # complex_feedback_propsed_codebook_model_savefname = './Saved Models/complex_feedback_propsed_{}_{}_beam_probing_codebook_{}_beam_noise_{}_dBm_tx_power_{}_dBm.pt'.format(dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm)
        # torch.save(complex_feedback_propsed_codebook_model.state_dict(),complex_feedback_propsed_codebook_model_savefname)    
        
        # plt.plot(val_acc,label='Proposed (complex)', linewidth=2)
        # plt.xlabel('Epoch', fontsize=14)  
        # plt.ylabel('Narrow beam pred. accuracy', fontsize=14)  

        # plt.xticks(fontsize=12) 
        # plt.yticks(fontsize=12) 
        # plt.title('Proposed probing beam (learned w/ narrow beam measurements)')

        # Proposed (magnitude)
        # magnitude_feedback_proposed_codebook_model = Beam_Classifier(n_antenna=n_antenna,n_wide_beam=n_wb,n_narrow_beam=n_nb,
        #                                         trainable_codebook=False,complex_codebook=magnitude_feedback_proposed_wb_codebook,
        #                                         noise_power=noise_power,norm_factor=norm_factor)
        # magnitude_feedback_proposed_codebook_opt = optim.Adam(magnitude_feedback_proposed_codebook_model.parameters(),lr=0.001, betas=(0.9,0.999), amsgrad=False)
        # train_loss_hist, val_loss_hist, train_acc, val_acc = fit(magnitude_feedback_proposed_codebook_model,top_k, train_loader, val_loader, magnitude_feedback_proposed_codebook_opt, nn.CrossEntropyLoss(), nepoch)
        # magnitude_feedback_proposed_codebook_model_savefname = './Saved Models/magnitude_feedback_propsed_{}_{}_beam_probing_codebook_{}_beam_noise_{}_dBm_tx_power_{}_dBm.pt'.format(dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm)
        # torch.save(magnitude_feedback_proposed_codebook_model.state_dict(),magnitude_feedback_proposed_codebook_model_savefname)    
        
        # plt.plot(val_acc,label='Proposed (magnitude)', linewidth=2)
        # plt.xlabel('Epoch', fontsize=14)  
        # plt.ylabel('Narrow beam pred. accuracy', fontsize=14)  
        # plt.xticks(fontsize=12) 
        # plt.yticks(fontsize=12)  
        # plt.title('Proposed probing beam (learned w/ narrow beam measurements)')

        
        # dft_wb_codebook = DFT_codebook(nseg=n_wb,n_antenna=n_antenna)
        # dft_codebook_cnn_model = Beam_Classifier(n_antenna=n_antenna,n_wide_beam=n_wb,n_narrow_beam=n_nb,
        #                             trainable_codebook=False,complex_codebook=dft_wb_codebook,
        #                             noise_power=noise_power,norm_factor=norm_factor)
        # dft_codebook_cnn_opt = optim.Adam(dft_codebook_cnn_model.parameters(),lr=0.01, betas=(0.9,0.999), amsgrad=False)
        # train_loss_hist, val_loss_hist, train_acc, val_acc = fit(dft_codebook_cnn_model,top_k, train_loader, val_loader, dft_codebook_cnn_opt, nn.CrossEntropyLoss(), nepoch)
        # dft_cnn_model_savefname = './Saved Models/DFTprob_{}_{}_beam_probing_codebook_{}_beam_noise_{}_dBm_tx_power_{}_dBm.pt'.format(dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm)
        # torch.save(dft_codebook_cnn_model.state_dict(),dft_cnn_model_savefname)    
        # plt.plot(val_acc,label='DFT probing beam', linewidth=2)
        
        # Trainable probing codebook + MLP classifier, MLP takes the received power as input
        learnable_model_savefname = './Saved Models/Ethan_{}_{}_beam_probing_codebook_{}_beam_noise_{}_dBm_tx_power_{}_dBm.pt'.format(dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm)
        directory_path = './Saved Codebooks'
        learnable_codebook_savefname = 'Ethan_{}_{}_beam_probing_codebook_{}_beam_noise_{}_dBm_tx_power_{}_dBm.npy'.format(dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm)
        learnable_codebook_save_path = os.path.join(directory_path, learnable_codebook_savefname)
        if os.path.exists(learnable_model_savefname) and os.path.exists(learnable_codebook_save_path) and not args.overwrite:
            print(f'[skip] Ethan artifacts exist for N_W={n_wb}, noise={noise_power_dBm}. Use --overwrite to retrain.')
            continue

        learnable_codebook_model = Beam_Classifier(n_antenna=n_antenna,n_wide_beam=n_wb,n_narrow_beam=n_nb,trainable_codebook=True,noise_power=noise_power,norm_factor=norm_factor).to(device)
        learnable_codebook_opt = optim.Adam(learnable_codebook_model.parameters(),lr=0.01, betas=(0.9,0.999), amsgrad=False)
        train_loss_hist, val_loss_hist, train_acc, val_acc = fit(learnable_codebook_model,top_k, train_loader_clean, val_loader_clean, learnable_codebook_opt, nn.CrossEntropyLoss(), nepoch)  
        learnable_codebook = learnable_codebook_model.get_codebook()
        np.save(learnable_codebook_save_path, learnable_codebook)
        torch.save(learnable_codebook_model.cpu().state_dict(),learnable_model_savefname)

                    
        plt.plot(val_acc,label='[10]', linewidth=2)
        plt.legend(fontsize=12)
        plt.title('{} probing beams {} narrow beams {} antennas'.format(n_wb,n_nb,n_antenna), fontsize=16)
        
        if not os.environ.get("MPLBACKEND") == "Agg":
            plt.show()

if args.train_only:
    raise SystemExit(0)

#%%  Test for mag-proposed with randomly phase rotated channel data
    
h_scaled_fake = np.multiply(h_scaled,np.exp(1j*np.random.uniform(-np.pi,np.pi,(h_scaled.shape[0],h_scaled.shape[1]))))
nb_measurement_complex_fake = np.matmul(h_scaled_fake, dft_nb_codebook.conj().T)

# nb_measurement_complex_fake =  np.multiply(nb_measurement_complex,np.exp(1j*np.random.uniform(-np.pi,np.pi,(nb_measurement_complex.shape[0],nb_measurement_complex.shape[1]))))
nb_measurement_complex_fake_concat = np.concatenate((np.real(nb_measurement_complex_fake),np.imag(nb_measurement_complex_fake)),axis=1)
x_test_fake2,y_test_fake2 = nb_measurement_complex_fake_concat[test_idc,:],nb_label[test_idc]

torch_x_test_fake2 = torch.from_numpy(x_test_fake2)
torch_y_test_fake2 = torch.from_numpy(y_test_fake2)

# Pytorch train and test sets



test_fake2 = torch.utils.data.TensorDataset(torch_x_test_fake2,torch_y_test_fake2)

# data loader
test_loader_fake2 = torch.utils.data.DataLoader(test_fake2, batch_size = batch_size, shuffle = False)    

if_fake=False
magnitude_feedback_projection_trainer_model2 = projection_trainer_fake(if_fake,n_wide_beam=n_wb,n_narrow_beam=n_nb,narrow_beam_CB=dft_nb_codebook,
                                            noise_power=noise_power,norm_factor=norm_factor)
magnitude_feedback_projection_trainer_model2_savefname = './Saved Projection Models/magnitude_feedback_projection_trainer_{}_{}_beam_probing_codebook_{}_beam_noise_{}_dBm_tx_power_{}_dBm.pt'.format(dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm)
# magnitude_feedback_projection_trainer_model2_savefname =  './Saved Projection Models/complex_feedback_projection_trainer_{}_{}_beam_probing_codebook_{}_beam_noise_{}_dBm_tx_power_{}_dBm.pt'.format(dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm)


magnitude_feedback_projection_trainer_model2.load_state_dict(torch.load(magnitude_feedback_projection_trainer_model2_savefname))

magnitude_feedback_trainer_acc = []
magnitude_feedback_trainer_topk_acc = []
magnitude_feedback_trainer_test_loss,magnitude_feedback_trainer_test_acc = eval_model(magnitude_feedback_projection_trainer_model2,test_loader_fake2,nn.CrossEntropyLoss()) 
magnitude_feedback_trainer_acc.append(magnitude_feedback_trainer_test_acc)
y_test_predict_magnitude_feedback_trainer = magnitude_feedback_projection_trainer_model2(torch_x_test_fake2.float()).detach().numpy()
topk_sorted_test_magnitude_feedback_trainer = (-y_test_predict_magnitude_feedback_trainer).argsort()
topk_acc_magnitude_feedback_trainer = []
for ue_bf_gain, pred_sort in zip(soft_label[test_idc,:],topk_sorted_test_magnitude_feedback_trainer):
    topk_acc = [ue_bf_gain.argmax() in pred_sort[:k] for k in range(1,4)]
    topk_acc_magnitude_feedback_trainer.append(topk_acc)
topk_acc_magnitude_feedback_trainer = np.array(topk_acc_magnitude_feedback_trainer).mean(axis=0)
magnitude_feedback_trainer_topk_acc.append(topk_acc_magnitude_feedback_trainer)
print(magnitude_feedback_trainer_topk_acc)

complex_feedback_projection_trainer_model2 = projection_trainer(n_wide_beam=n_wb,n_narrow_beam=n_nb,narrow_beam_CB=dft_nb_codebook,
                                            noise_power=noise_power,norm_factor=norm_factor)
complex_feedback_projection_trainer_model2_savefname = './Saved Projection Models/complex_feedback_projection_trainer_{}_{}_beam_probing_codebook_{}_beam_noise_{}_dBm_tx_power_{}_dBm.pt'.format(dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm)
# magnitude_feedback_projection_trainer_model2_savefname =  './Saved Projection Models/complex_feedback_projection_trainer_{}_{}_beam_probing_codebook_{}_beam_noise_{}_dBm_tx_power_{}_dBm.pt'.format(dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm)


complex_feedback_projection_trainer_model2.load_state_dict(torch.load(complex_feedback_projection_trainer_model2_savefname))

complex_feedback_trainer_acc = []
complex_feedback_trainer_topk_acc = []
complex_feedback_trainer_test_loss,complex_feedback_trainer_test_acc = eval_model(complex_feedback_projection_trainer_model2,test_loader_fake2,nn.CrossEntropyLoss()) 
complex_feedback_trainer_acc.append(complex_feedback_trainer_test_acc)
y_test_predict_complex_feedback_trainer = complex_feedback_projection_trainer_model2(torch_x_test_fake2.float()).detach().numpy()
topk_sorted_test_complex_feedback_trainer = (-y_test_predict_complex_feedback_trainer).argsort()
topk_acc_complex_feedback_trainer = []
for ue_bf_gain, pred_sort in zip(soft_label[test_idc,:],topk_sorted_test_complex_feedback_trainer):
    topk_acc = [ue_bf_gain.argmax() in pred_sort[:k] for k in range(1,4)]
    topk_acc_complex_feedback_trainer.append(topk_acc)
topk_acc_complex_feedback_trainer = np.array(topk_acc_complex_feedback_trainer).mean(axis=0)
complex_feedback_trainer_topk_acc.append(topk_acc_complex_feedback_trainer)
print(complex_feedback_trainer_topk_acc)

#%%
n_wb=16
n_nb=64
noise_power_dBm=-81
top_k=1

n_antenna=64

dataset_name = 'O1_28B_ULA'

for i in [0,1,3]:
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
        complex_feed_proposed_probing_codebook = np.load('./Saved Codebooks/complex_feedback_proposed_codebook_{}_{}_beam_probing_codebook_{}_beam_noise_{}_dBm_tx_power_{}_dBm.npy'.format(dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm))
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
        magnitude_feed_proposed_probing_codebook = np.load('./Saved Codebooks/magnitude_feedback_proposed_codebook_{}_{}_beam_probing_codebook_{}_beam_noise_{}_dBm_tx_power_{}_dBm.npy'.format(dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm))
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

