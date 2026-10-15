#%%

# os.chdir('/root/bkup/Learning_site_specific')
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from ComplexLayers_Torch import Beam_Classifier, projection_trainer,projection_trainer_fake,  fit,fit_projector,  eval_model, custom_projection_L
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
from beam_utils_L import ULA_DFT_codebook as DFT_codebook
from beam_utils_L import ULA_DFT_codebook_blockmatrix as DFT_codebook_blockmatrix
from beam_utils_L import plot_codebook_pattern, codebook_blockmatrix, get_AMCF_codebook, calc_beam_pattern, generate_phase_noise_matrix, generate_noise_matrix
np.random.seed(8)

n_wb=12
n_nb=64
noise_power_dBm=-81
top_k=1

n_antenna=64

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
        ax1.set_title('Learned w/ channel knowledge')
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
        ax3.set_title('Proposed')
        fig3.tight_layout()