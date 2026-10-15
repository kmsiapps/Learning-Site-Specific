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
from ComplexLayers_Torch import Beam_Classifier, projection_trainer,projection_trainer_classifieronly,  fit,fit_projector,  eval_model, custom_projection_L
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
from beam_utils_L import ULA_DFT_codebook as DFT_codebook
from beam_utils_L import ULA_DFT_codebook_blockmatrix as DFT_codebook_blockmatrix
from beam_utils_L import plot_codebook_pattern, codebook_blockmatrix, get_AMCF_codebook, calc_beam_pattern, generate_phase_noise_matrix, generate_noise_matrix
np.random.seed(8)
# number of probing beams (N_W)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n_wide_beams",
        type=int,
        nargs="+",
        default=[16],
        help="List of probing beams (e.g., --n_wide_beams 16 32)"
    )

    parser.add_argument(
        "--noise_power_dBm_list",
        type=float,
        nargs="+",
        default=[-81],
        help="List of noise power in dBm (e.g., --noise_power_dBm_list -81 -90)"
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="O1_28B_ULA",
        choices=["O1_28B_ULA", "Rosslyn_ULA"],
        help="Dataset name"
    )
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=500, help="Training batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Training device")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing checkpoints")

    return parser.parse_args()


args = parse_args()

n_wide_beams = args.n_wide_beams
noise_power_dBm_list = args.noise_power_dBm_list
dataset_name = args.dataset_name
device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

print('=' * 50)
print(f"Probing beams (N_W): {n_wide_beams}")
print(f"Noise power (dBm): {noise_power_dBm_list}")
print(f"Dataset name: {dataset_name}")
print(f"Device: {device}")

# n_wide_beams = [16, 32]
# number of narrow data beams to select from (N_V)
n_narrow_beams = [64 for i in n_wide_beams]

n_antenna = 64
top_k=1
nepoch = args.epochs
batch_size = args.batch_size            
noise_factor = 0 #dB
# noise_power_dBm_list = [-81, -90] 
noiseless = False

# dataset_name = 'O1_28B_ULA' # 'Rosslyn_ULA' or 'O1_28B_ULA' or 'I3_60_ULA' or 'O1_28_ULA'
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
        pin_memory = device.type == "cuda"
        loader_kwargs = dict(batch_size=batch_size, shuffle=False, pin_memory=pin_memory,
                             num_workers=args.num_workers, persistent_workers=args.num_workers > 0)
        train_loader = torch.utils.data.DataLoader(train, **loader_kwargs)
        val_loader = torch.utils.data.DataLoader(val, **loader_kwargs)
        test_loader = torch.utils.data.DataLoader(test, **loader_kwargs)
        
        dft_cnn_model_savefname = './Saved Models/DFTprob_{}_{}_beam_probing_codebook_{}_beam_noise_{}_dBm_tx_power_{}_dBm.pt'.format(dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm)
        if os.path.exists(dft_cnn_model_savefname) and not args.overwrite:
            print(f'[skip] {dft_cnn_model_savefname} exists. Use --overwrite to retrain.')
            continue

        dft_wb_codebook = DFT_codebook(nseg=n_wb,n_antenna=n_antenna)
        dft_codebook_cnn_model = Beam_Classifier(n_antenna=n_antenna,n_wide_beam=n_wb,n_narrow_beam=n_nb,
                                    trainable_codebook=False,complex_codebook=dft_wb_codebook,
                                    noise_power=noise_power,norm_factor=norm_factor).to(device)
        dft_codebook_cnn_opt = optim.Adam(dft_codebook_cnn_model.parameters(),lr=0.01, betas=(0.9,0.999), amsgrad=False)
        train_loss_hist, val_loss_hist, train_acc, val_acc = fit(dft_codebook_cnn_model,top_k, train_loader, val_loader, dft_codebook_cnn_opt, nn.CrossEntropyLoss(), nepoch)
        torch.save(dft_codebook_cnn_model.cpu().state_dict(),dft_cnn_model_savefname)    
        # plt.plot(val_acc,label='DFT probing beam', linewidth=2)
        