#%%
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 23:04:53 2021

@author: Yuqiang (Ethan) Heng
"""
os.chdir('/root/bkup/Learning_site_specific')

import time
import numpy as np
import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from beam_utils_L import ULA_DFT_codebook as DFT_codebook
import seaborn as sns

np.random.seed(7)
# number of narrow data beams to select from (N_V)
n_nb = 64
# number of probing beams (N_W)
n_wide_beams =[16]
n_antenna = 64
antenna_sel = np.arange(n_antenna)
tsne_perplexity = 30
noise_power_dBm = -81
top_k=1

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
    tx_power_dBm=10
elif dataset_name == 'O1_28B_ULA':
    h = np.load('./Dataset/test_O1_28B_BS1x64x1_UE1x1x1.npy')
    h=h.squeeze()
    h_real = np.real(h)
    h_imag = np.imag(h)
    tx_power_dBm=20
elif dataset_name == 'I3_60_ULA':
    fname_h_real = './Dataset/DeepMIMO Dataset/I3_60_ULA/h_real.mat'
    fname_h_imag = './Dataset/DeepMIMO Dataset/I3_60_ULA/h_imag.mat'
    h_real = sio.loadmat(fname_h_real)['h_real']
    h_imag = sio.loadmat(fname_h_imag)['h_imag']
elif dataset_name == 'O1_28_ULA':
    fname_h_real = './Dataset/DeepMIMO Dataset/O1_28_ULA/h_real.mat'
    fname_h_imag = './Dataset/DeepMIMO Dataset/O1_28_ULA/h_imag.mat'
    h_real = sio.loadmat(fname_h_real)['h_real']
    h_imag = sio.loadmat(fname_h_imag)['h_imag']
else:
    raise NameError('Dataset Not Supported')

norm_factor = np.max(abs(h))
h_scaled = h/norm_factor
train_idc, test_idc = train_test_split(np.arange(h.shape[0]),test_size=0.2)
val_idc, test_idc = train_test_split(test_idc,test_size=0.5)
dft_nb_codebook = DFT_codebook(nseg=n_nb,n_antenna=n_antenna)
label = np.argmax(np.power(np.absolute(np.matmul(h_scaled, dft_nb_codebook.T)),2),axis=1)


x_train,y_train = h_scaled[train_idc,:],label[train_idc]
x_test,y_test = h_scaled[test_idc,:],label[test_idc]
nb_measurement_train = np.absolute(np.matmul(h_scaled[train_idc,:],dft_nb_codebook.T))
nb_measurement_test = np.absolute(np.matmul(h_scaled[test_idc,:],dft_nb_codebook.T))

for i,n_wb in enumerate(n_wide_beams):  

    trainable_codebook = np.load('./Saved Codebooks/Ethan_{}_{}_beam_probing_codebook_{}_beam_noise_{}_dBm_tx_power_{}_dBm.npy'.format(dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm)).T
    dft_codebook = DFT_codebook(nseg=n_wb,n_antenna=n_antenna)
    complex_feedback_proposed_codebook = np.load('./Saved Codebooks/complex_feedback_proposed_codebook_{}_{}_beam_probing_codebook_{}_beam_noise_{}_dBm_tx_power_{}_dBm.npy'.format(dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm))
    magnitude_feedback_proposed_codebook = np.load('./Saved Codebooks/magnitude_feedback_proposed_codebook_{}_{}_beam_probing_codebook_{}_beam_noise_{}_dBm_tx_power_{}_dBm.npy'.format(dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm))
    

    magnitude_feedback_projection_matrix_savefname = './Saved Projections/magnitude_feedback_projection_matrix_{}_{}_beam_probing_codebook_{}_beam_noise_{}_dBm_tx_power_{}_dBm.npy'.format(dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm)
    # magnitude_feedback_projection_matrix = np.load(magnitude_feedback_projection_matrix_savefname)/ np.sqrt(n_antenna)
    
    magnitude_feedback_projection_matrix_blockmatrix = np.load(magnitude_feedback_projection_matrix_savefname)
    magnitude_feedback_projection_matrix = magnitude_feedback_projection_matrix_blockmatrix[:magnitude_feedback_projection_matrix_blockmatrix.shape[0]//2,:magnitude_feedback_projection_matrix_blockmatrix.shape[1]//2]+1j*magnitude_feedback_projection_matrix_blockmatrix[magnitude_feedback_projection_matrix_blockmatrix.shape[0]//2:,magnitude_feedback_projection_matrix_blockmatrix.shape[1]//2:]
    magnitude_feedback_projection_matrix = magnitude_feedback_projection_matrix / np.sqrt(n_antenna)
    
    feat_cols = ['beam_{}'.format(bi) for bi in range(n_wb)]
    # h_project_trainable = np.power(np.absolute(np.matmul(x_test, trainable_codebook.conj().T)),2)
    # df_trainable = pd.DataFrame(h_project_trainable,columns=feat_cols)
    # df_trainable['y'] = y_test
    # df_trainable['label'] = df_trainable['y'].apply(lambda i: str(i))

    # time_start = time.time()
    # tsne_trainable = TSNE(n_components=2, verbose=0, perplexity=tsne_perplexity, n_iter=5000)
    # trainable_tsne_results = tsne_trainable.fit_transform(df_trainable[feat_cols].values)
    # print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    # df_trainable['t-SNE Dim 1'] = trainable_tsne_results[:,0]
    # df_trainable['t-SNE Dim 2'] = trainable_tsne_results[:,1]
    

    # h_project_dft = np.power(np.absolute(np.matmul(x_test, dft_codebook.conj().T)),2)
    # df_dft = pd.DataFrame(h_project_dft,columns=feat_cols)
    # df_dft['y'] = y_test
    # df_dft['label'] = df_dft['y'].apply(lambda i: str(i))

    # time_start = time.time()
    # tsne_dft = TSNE(n_components=2, verbose=0, perplexity=tsne_perplexity, n_iter=5000)
    # dft_tsne_results = tsne_dft.fit_transform(df_dft[feat_cols].values)
    # print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    # df_dft['t-SNE Dim 1'] = dft_tsne_results[:,0]
    # df_dft['t-SNE Dim 2'] = dft_tsne_results[:,1]    
    
    # h_project_unconstrained = np.power(np.absolute(np.matmul(x_test, complex_feedback_proposed_codebook.conj().T)),2)
    # df_unconstrained = pd.DataFrame(h_project_unconstrained,columns=feat_cols)
    # df_unconstrained['y'] = y_test
    # df_unconstrained['label'] = df_unconstrained['y'].apply(lambda i: str(i))

    # time_start = time.time()
    # tsne_unconstrained = TSNE(n_components=2, verbose=0, perplexity=tsne_perplexity, n_iter=5000)
    # unconstrained_tsne_results = tsne_unconstrained.fit_transform(df_unconstrained[feat_cols].values)
    # print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    # df_unconstrained['t-SNE Dim 1'] = unconstrained_tsne_results[:,0]
    # df_unconstrained['t-SNE Dim 2'] = unconstrained_tsne_results[:,1]   
    
    phase_random = np.exp(1j*np.random.uniform(-np.pi,np.pi,(nb_measurement_test.shape[0],nb_measurement_test.shape[1])))

    h_random = np.power(np.absolute(np.matmul(np.multiply(nb_measurement_test,phase_random), magnitude_feedback_projection_matrix.T)),2)
    
    df_random = pd.DataFrame(h_random,columns=feat_cols)
    df_random['y'] = y_test
    df_random['label'] = df_random['y'].apply(lambda i: str(i))

    time_start = time.time()
    tsne_random = TSNE(n_components=2, verbose=0, perplexity=tsne_perplexity, n_iter=5000)
    random_tsne_results = tsne_random.fit_transform(df_random[feat_cols].values)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    df_random['t-SNE Dim 1'] = random_tsne_results[:,0]
    df_random['t-SNE Dim 2'] = random_tsne_results[:,1]   
    

    # fig1,ax1 = plt.subplots(figsize=(4.5,4.5))
    # sns.scatterplot(
    #     x='t-SNE Dim 1', y='t-SNE Dim 2',
    #     hue="y",
    #     palette=sns.color_palette("hls", len(df_trainable['y'].unique())),
    #     data=df_trainable,
    #     legend=False,
    #     alpha=0.3,
    #     ax=ax1
    # )    
    # ax1.set_xticks(np.arange(-150,200,50))
    # ax1.set_yticks(np.arange(-150,200,50))
    
    # fig2,ax2 = plt.subplots(figsize=(4.5,4.5))
    # sns.scatterplot(
    #     x='t-SNE Dim 1', y='t-SNE Dim 2',
    #     hue="y",
    #     palette=sns.color_palette("hls", len(df_dft['y'].unique())),
    #     data=df_dft,
    #     legend=False,
    #     alpha=0.3,
    #     ax=ax2
    # )   
    # ax2.set_xticks(np.arange(-150,200,50))
    # ax2.set_yticks(np.arange(-150,200,50))


    # fig3,ax3 = plt.subplots(figsize=(4.5,4.5))
    # sns.scatterplot(
    #     x='t-SNE Dim 1', y='t-SNE Dim 2',
    #     hue="y",
    #     palette=sns.color_palette("hls", len(df_unconstrained['y'].unique())),
    #     data=df_unconstrained,
    #     legend=False,
    #     alpha=0.3,
    #     ax=ax3
    # )   
    # ax3.set_xticks(np.arange(-150,200,50))
    # ax3.set_yticks(np.arange(-150,200,50))
    
    fig4,ax4 = plt.subplots(figsize=(4.5,4.5))
    sns.scatterplot(
        x='t-SNE Dim 1', y='t-SNE Dim 2',
        hue="y",
        palette=sns.color_palette("hls", len(df_random['y'].unique())),
        data=df_random,
        legend=False,
        alpha=0.3,
        ax=ax4
    )   
    ax4.set_xticks(np.arange(-150,200,50))
    ax4.set_yticks(np.arange(-150,200,50))
    

# %%
