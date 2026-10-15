# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 20:04:13 2021

@author: Yuqiang (Ethan) Heng
"""
#%%
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from ComplexLayers_Torch import Beam_Classifier, eval_model, Beam_Classifier_CNN
import torch.utils.data
import torch.nn as nn
from sklearn.model_selection import train_test_split
from beam_utils_L import ULA_DFT_codebook as DFT_codebook
from beam_utils_L import plot_codebook_pattern, DFT_angles, Beam_Search_Tree, get_AMCF_codebook, AMCF_boundaries
from beam_utils_L import codebook_blockmatrix
np.random.seed(7)
# number of probing beams (N_W)
n_wide_beams = [8]
# number of narrow data beams to select from (N_V)
n_narrow_beams = [64 for i in n_wide_beams]
n_antenna = 64
batch_size = 500

noise_power_dBm = -81
noise_factor = 0 #dB

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

norm_factor = np.max(abs(h))
h_scaled = h/norm_factor
h_concat_scaled = np.concatenate((h_real/norm_factor,h_imag/norm_factor),axis=1)

train_idc, test_idc = train_test_split(np.arange(h.shape[0]),test_size=0.4)
val_idc, test_idc = train_test_split(test_idc,test_size=0.5)



cb_noise_list = [0,0.001,0.01,0.05,0.1,0.15,0.2]
learned_snr_cb_noise_list = np.zeros((3,len(cb_noise_list)))
proposed_snr_cb_noise_list = np.zeros((3,len(cb_noise_list)))
learned_acc_cb_noise_list = np.zeros((3,len(cb_noise_list)))
proposed_acc_cb_noise_list = np.zeros((3,len(cb_noise_list)))
for n_wb_i, n_wb in enumerate(n_wide_beams):
    n_nb = n_narrow_beams[n_wb_i]
    print('{} Wide Beams, {} Narrow Beams.'.format(n_wb,n_nb))
    for cb_noise_i, cb_noise in enumerate(cb_noise_list):
        print('{} cb noise.'.format(cb_noise))    
        
        if cb_noise ==0:
            distortion_trial = 1
        else:
            distortion_trial = 10


        dft_codebook_acc = []
        learnable_codebook_acc = []
        proposed_codebook_acc = []


        dft_codebook_topk_acc = []
        learnable_codebook_topk_acc = []
        proposed_codebook_topk_acc = []

        dft_codebook_topk_gain = []
        learned_codebook_topk_gain= []
        proposed_codebook_topk_gain= []

        optimal_gains = []

        for distortion_iter in range(0,distortion_trial):
            
            V_noise_savefile_name = 'additive_V_noise_{}_projection_L_{}_{}_beam_probing_codebook_{}_beam_complex_input_CNN_classifier_noise_{}_dBm_tx_power_{}_dBm_distortion_iter_{}.npy'.format(cb_noise,dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm,distortion_iter)
            V_noise = np.load(V_noise_savefile_name)


            W_noise_savefile_name = 'additive_W_noise_{}_projection_L_{}_{}_beam_probing_codebook_{}_beam_complex_input_CNN_classifier_noise_{}_dBm_tx_power_{}_dBm_distortion_iter_{}_RSRP.npy'.format(cb_noise,dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm,distortion_iter)
            W_noise = np.load(W_noise_savefile_name).T

            learnable_codebook =np.load('./Saved Codebooks/Top{}_{}_trainable_{}_beam_probing_codebook_{}_beam_complex_input_CNN_classifier_noise_{}_dBm_tx_power_{}_dBm_RSRP.npy'.format(top_k,dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm))
            proposed_codebook = np.load('./Saved Codebooks/additive_distorted_{}_projection_L_{}_{}_beam_probing_codebook_{}_beam_complex_input_CNN_classifier_noise_{}_dBm_tx_power_{}_dBm_distortion_iter_{}.npy'.format(cb_noise,dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm,distortion_iter))
            
            
            # distorted_learnable_codebook = np.multiply(learnable_codebook,  W_noise)
            distorted_learnable_codebook = learnable_codebook + W_noise
            

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
            learnable_model_savefname = './Saved Models/Top{}_{}_trainable_{}_beam_probing_codebook_{}_beam_complex_input_CNN_classifier_noise_{}_dBm_tx_power_{}_dBm_RSRP.pt'.format(top_k,dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm)
            learnable_codebook_model.load_state_dict(torch.load(learnable_model_savefname))
            
            learnable_codebook_model.codebook.reset_parameters(theta=torch.Tensor(np.angle(distorted_learnable_codebook)))

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
            # dft_codebook_model = Beam_Classifier_CNN(n_antenna=n_antenna,n_wide_beam=n_wb,n_narrow_beam=n_nb,
            #                                     trainable_codebook=False,complex_codebook=dft_wb_codebook,
            #                                     noise_power=noise_power,norm_factor=norm_factor)
            # dft_model_savefname = './Saved Models/DFTprob_Top{}_{}_DFT_{}_beam_probing_codebook_{}_beam_complex_input_CNN_classifier_noise_{}_dBm_tx_power_{}_dBm.pt'.format(top_k,dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm)
            # dft_codebook_model.load_state_dict(torch.load(dft_model_savefname))
            # dft_codebook_test_loss,dft_codebook_test_acc = eval_model(dft_codebook_model,test_loader,nn.CrossEntropyLoss()) 
            # dft_codebook_acc.append(dft_codebook_test_acc)
            # y_test_predict_dft_codebook = dft_codebook_model(torch_x_test.float()).detach().numpy()
            # topk_sorted_test_dft_codebook = (-y_test_predict_dft_codebook).argsort()
            # topk_bf_gain_dft_codebook = []
            # topk_acc_dft_codebook = []
            # for ue_bf_gain, pred_sort in zip(soft_label[test_idc,:],topk_sorted_test_dft_codebook):
            #     topk_gains = [ue_bf_gain[pred_sort[:k]].max() for k in range(1,4)]
            #     topk_bf_gain_dft_codebook.append(topk_gains)
            #     topk_acc = [ue_bf_gain.argmax() in pred_sort[:k] for k in range(1,4)]
            #     topk_acc_dft_codebook.append(topk_acc)        
            # topk_bf_gain_dft_codebook = np.array(topk_bf_gain_dft_codebook)
            # dft_codebook_topk_gain.append(topk_bf_gain_dft_codebook)
            # topk_acc_dft_codebook = np.array(topk_acc_dft_codebook).mean(axis=0)
            # dft_codebook_topk_acc.append(topk_acc_dft_codebook)

            # dft_nb_codebook = DFT_codebook(nseg=n_nb,n_antenna=n_antenna)

            # projection_L_savefname = 'distorted_{}_Top{}_projection_L_{}_{}_beam_probing_codebook_{}_beam_complex_input_CNN_classifier_noise_{}_dBm_tx_power_{}_dBm.npy'.format(cb_noise,top_k,dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm)
            # unconstrained_wb_codebook = np.load(projection_L_savefname)
            # unconstrained_wb_codebook = unconstrained_wb_codebook[:unconstrained_wb_codebook.shape[0]//2,:unconstrained_wb_codebook.shape[1]//2]+1j*unconstrained_wb_codebook[unconstrained_wb_codebook.shape[0]//2:,unconstrained_wb_codebook.shape[1]//2:]
            # unconstrained_wb_codebook = np.matmul(unconstrained_wb_codebook,dft_nb_codebook)
            # for i in range(0,unconstrained_wb_codebook.shape[0]):
            #     unconstrained_wb_codebook[i,:] = unconstrained_wb_codebook[i,:]/np.sqrt(np.sum(np.abs(unconstrained_wb_codebook[i,:]**2)))
            
            # normalized_unconstrained_wb_codebook = unconstrained_wb_codebook
            # normalized_unconstrained_wb_codebook = np.multiply(normalized_unconstrained_wb_codebook,W_noise.T)
            proposed_codebook_model = Beam_Classifier(n_antenna=n_antenna,n_wide_beam=n_wb,n_narrow_beam=n_nb,
                                                trainable_codebook=False,complex_codebook=proposed_codebook,
                                                noise_power=noise_power,norm_factor=norm_factor)
            proposed_model_savefname = './Saved Models/additive_distorted_{}_Top{}_{}_DFT_{}_beam_probing_codebook_{}_beam_complex_input_CNN_classifier_noise_{}_dBm_tx_power_{}_dBm_distortion_iter_{}_RSRP.pt'.format(cb_noise,top_k,dataset_name,n_wb,n_nb,noise_power_dBm,tx_power_dBm,distortion_iter)
            proposed_codebook_model.load_state_dict(torch.load(proposed_model_savefname))
            proposed_codebook_test_loss,proposed_codebook_test_acc = eval_model(proposed_codebook_model,test_loader,nn.CrossEntropyLoss()) 
            proposed_codebook_acc.append(proposed_codebook_test_acc)
            y_test_predict_proposed_codebook = proposed_codebook_model(torch_x_test.float()).detach().numpy()
            topk_sorted_test_proposed_codebook = (-y_test_predict_proposed_codebook).argsort()
            topk_bf_gain_proposed_codebook = []
            topk_acc_proposed_codebook = []
            for ue_bf_gain, pred_sort in zip(soft_label[test_idc,:],topk_sorted_test_proposed_codebook):
                topk_gains = [ue_bf_gain[pred_sort[:k]].max() for k in range(1,4)]
                topk_bf_gain_proposed_codebook.append(topk_gains)
                topk_acc = [ue_bf_gain.argmax() in pred_sort[:k] for k in range(1,4)]
                topk_acc_proposed_codebook.append(topk_acc)
            topk_bf_gain_proposed_codebook = np.array(topk_bf_gain_proposed_codebook)
            proposed_codebook_topk_gain.append(topk_bf_gain_proposed_codebook)
            topk_acc_proposed_codebook = np.array(topk_acc_proposed_codebook).mean(axis=0)
            proposed_codebook_topk_acc.append(topk_acc_proposed_codebook)
            
            optimal_gains.append(soft_label[test_idc,:].max(axis=-1))

        # dft_codebook_topk_gain = np.array(dft_codebook_topk_gain)
        learned_codebook_topk_gain = np.array(learned_codebook_topk_gain)
        proposed_codebook_topk_gain = np.array(proposed_codebook_topk_gain)
        optimal_gains = np.array(optimal_gains)

        # dft_codebook_topk_snr = tx_power_dBm + 10*np.log10(dft_codebook_topk_gain) - noise_power_dBm + noise_factor
        learned_codebook_topk_snr = tx_power_dBm + 10*np.log10(learned_codebook_topk_gain) - noise_power_dBm + noise_factor
        proposed_codebook_topk_snr = tx_power_dBm + 10*np.log10(proposed_codebook_topk_gain) - noise_power_dBm + noise_factor

        optimal_snr = tx_power_dBm + 10*np.log10(optimal_gains) - noise_power_dBm + noise_factor


        # dft_codebook_topk_acc = np.array(dft_codebook_topk_acc)
        learnable_codebook_topk_acc = np.array(learnable_codebook_topk_acc)
        proposed_codebook_topk_acc = np.array(proposed_codebook_topk_acc)

        len_h = learned_codebook_topk_snr.shape[1]
        learned_codebook_topk_snr = learned_codebook_topk_snr.reshape(distortion_trial*len_h,3).mean(axis=0) 
        proposed_codebook_topk_snr = proposed_codebook_topk_snr.reshape(distortion_trial*len_h,3).mean(axis=0) 
        
        learnable_codebook_topk_acc = learnable_codebook_topk_acc.mean(axis=0)
        proposed_codebook_topk_acc = proposed_codebook_topk_acc.mean(axis=0)
        """    
        compute exhaustive beam search acc and snr
        """
        # dft_nb_codebook = DFT_codebook(nseg=64,n_antenna=64)
        
        # dft_nb_az = DFT_angles(64)
        # dft_nb_az = np.arcsin(1/0.5*dft_nb_az)
        # nb_bf_signal = np.matmul(h[test_idc], dft_nb_codebook.conj().T)
        # nb_bf_noise_real = np.random.normal(loc=0,scale=1,size=nb_bf_signal.shape)*np.sqrt(noise_power/2)
        # nb_bf_noise_imag = np.random.normal(loc=0,scale=1,size=nb_bf_signal.shape)*np.sqrt(noise_power/2)
        # nb_bf_signal_with_noise = nb_bf_signal + nb_bf_noise_real + 1j*nb_bf_noise_imag
        # nb_bf_gain = np.power(np.absolute(nb_bf_signal),2)
        # nb_bf_gain_with_noise = np.power(np.absolute(nb_bf_signal_with_noise),2)
        # best_nb_noisy = np.argmax(nb_bf_gain_with_noise,axis=1)
        # best_nb = np.argmax(nb_bf_gain,axis=1)
        # best_nb_az = dft_nb_az[best_nb]
        # exhaustive_acc = (best_nb_noisy==best_nb).mean()
        # genie_nb_snr = tx_power_dBm + 10*np.log10(nb_bf_gain.max(axis=1)) - noise_power_dBm + noise_factor
        # exhaustive_nb_snr = np.array([nb_bf_gain[ue_idx,best_nb_idx_noisy] for ue_idx,best_nb_idx_noisy in enumerate(best_nb_noisy)])
        # exhaustive_nb_snr = tx_power_dBm + 10*np.log10(exhaustive_nb_snr) - noise_power_dBm + noise_factor
        # print('Genie avg. SNR = {} dB.'.format(genie_nb_snr.mean()))
        # print('Exhaustive search accuracy = {}, avg. SNR = {} dB.'.format(exhaustive_acc,exhaustive_nb_snr.mean()))



        "Opt narrow beam prediction accuracy"
        plt.figure(figsize=(6,4.5))
        plt.plot(n_wide_beams,learnable_codebook_topk_acc[2],'--',marker='s',label='[10], k={}'.format(3))
        plt.plot(n_wide_beams,learnable_codebook_topk_acc[1],'-.',marker='s',label='[10], k={}'.format(2))
        plt.plot(n_wide_beams,learnable_codebook_topk_acc[0],marker='s',label='[10], k={}'.format(1))
        plt.hlines(y=exhaustive_acc,xmin=min(n_wide_beams),xmax=max(n_wide_beams),linestyles='dotted',label='Exhaustive search')
        plt.xticks(n_wide_beams)
        plt.legend()
        plt.xlabel('probing codebook size')
        plt.ylabel('Accuracy')
        plt.title('Optimal narrow beam prediction accuracy_cbnoise_{}'.format(cb_noise))
        plt.show()
        


        # plt.figure(figsize=(6,4.5))
        # plt.plot(n_wide_beams,dft_codebook_topk_acc[:,2],'--',marker='s',label='DFT, k={}'.format(3))
        # plt.plot(n_wide_beams,dft_codebook_topk_acc[:,1],'-.',marker='s',label='DFT, k={}'.format(2))
        # plt.plot(n_wide_beams,dft_codebook_topk_acc[:,0],marker='s',label='DFT, k={}'.format(1))
        # plt.hlines(y=exhaustive_acc,xmin=min(n_wide_beams),xmax=max(n_wide_beams),linestyles='dotted',label='Exhaustive search')
        # plt.xticks(n_wide_beams)
        # plt.legend()
        # plt.xlabel('probing codebook size')
        # plt.ylabel('Accuracy')
        # plt.title('Optimal narrow beam prediction accuracy')
        # plt.show()

        plt.figure(figsize=(6,4.5))
        plt.plot(n_wide_beams,proposed_codebook_topk_acc[2],'--',marker='s',label='Proposed method, k={}'.format(3))
        plt.plot(n_wide_beams,proposed_codebook_topk_acc[1],'-.',marker='s',label='Proposed method, k={}'.format(2))
        plt.plot(n_wide_beams,proposed_codebook_topk_acc[0],marker='s',label='Proposed method, k={}'.format(1))
        plt.hlines(y=exhaustive_acc,xmin=min(n_wide_beams),xmax=max(n_wide_beams),linestyles='dotted',label='Exhaustive search')
        plt.xticks(n_wide_beams)
        plt.legend()
        plt.xlabel('probing codebook size')
        plt.ylabel('Accuracy')
        plt.title('Optimal narrow beam prediction accuracy_cbnoise_{}'.format(cb_noise))
        plt.show()

        "Average SNRs"
        plt.figure(figsize=(6,4.5))
        plt.plot(n_wide_beams,learned_codebook_topk_snr[2],'--',marker='s',label='[10], k={}'.format(3))
        plt.plot(n_wide_beams,learned_codebook_topk_snr[1],'-.',marker='s',label='[10], k={}'.format(2))
        plt.plot(n_wide_beams,learned_codebook_topk_snr[0],marker='s',label='[10], k={}'.format(1))
        plt.xticks(n_wide_beams)
        plt.legend()
        plt.xlabel('probing codebook size')
        plt.ylabel('Average SNR (dB)')
        plt.title('SNR of top-k predicted beams_cbnoise_{}'.format(cb_noise))
        plt.show()

        # plt.figure(figsize=(6,4.5))
        # plt.plot(n_wide_beams,dft_codebook_topk_snr[:,:,2].mean(axis=1),'--',marker='s',label='DFT, k={}'.format(3))
        # plt.plot(n_wide_beams,dft_codebook_topk_snr[:,:,1].mean(axis=1),'-.',marker='s',label='DFT, k={}'.format(2))
        # plt.plot(n_wide_beams,dft_codebook_topk_snr[:,:,0].mean(axis=1),marker='s',label='DFT, k={}'.format(1))
        # plt.xticks(n_wide_beams)
        # plt.legend()
        # plt.xlabel('probing codebook size')
        # plt.ylabel('Average SNR (dB)')
        # plt.title('SNR of top-k predicted beams')
        # plt.show()

        plt.figure(figsize=(6,4.5))
        plt.plot(n_wide_beams,proposed_codebook_topk_snr[2],'--',marker='s',label='Proposed, k={}'.format(3))
        plt.plot(n_wide_beams,proposed_codebook_topk_snr[1],'-.',marker='s',label='Proposed, k={}'.format(2))
        plt.plot(n_wide_beams,proposed_codebook_topk_snr[0],marker='s',label='Proposed, k={}'.format(1))
        plt.xticks(n_wide_beams)
        plt.legend()
        plt.xlabel('probing codebook size')
        plt.ylabel('Average SNR (dB)')
        plt.title('SNR of top-k predicted beams_cbnoise_{}'.format(cb_noise))
        plt.show()

        # for i,N in enumerate(n_wide_beams):     
        #     fig,ax = plot_codebook_pattern(learned_codebooks[i].T)
        #     ax.set_title('Trainable {}-Beam Codebook'.format(N))
        #     fig,ax = plot_codebook_pattern(dft_codebooks[i])
        #     ax.set_title('DFT {}-Beam Codebook'.format(N))
        #     fig,ax = plot_codebook_pattern(AMCF_codebooks[i])
        #     ax.set_title('AMCF {}-Beam Codebook'.format(N))
        learned_snr_cb_noise_list[2,cb_noise_i] = learned_codebook_topk_snr[2]
        learned_snr_cb_noise_list[1,cb_noise_i] = learned_codebook_topk_snr[1]
        learned_snr_cb_noise_list[0,cb_noise_i] = learned_codebook_topk_snr[0]
        proposed_snr_cb_noise_list[2,cb_noise_i] = proposed_codebook_topk_snr[2]
        proposed_snr_cb_noise_list[1,cb_noise_i] = proposed_codebook_topk_snr[1]
        proposed_snr_cb_noise_list[0,cb_noise_i] = proposed_codebook_topk_snr[0]

        learned_acc_cb_noise_list[2,cb_noise_i] = learnable_codebook_topk_acc[2]
        learned_acc_cb_noise_list[1,cb_noise_i] = learnable_codebook_topk_acc[1]
        learned_acc_cb_noise_list[0,cb_noise_i] = learnable_codebook_topk_acc[0]
        proposed_acc_cb_noise_list[2,cb_noise_i] = proposed_codebook_topk_acc[2]
        proposed_acc_cb_noise_list[1,cb_noise_i] = proposed_codebook_topk_acc[1]
        proposed_acc_cb_noise_list[0,cb_noise_i] = proposed_codebook_topk_acc[0]
        #############################################################################

        # preivious one 
    
        # "Opt narrow beam prediction accuracy"
        # plt.figure(figsize=(6,4.5))
        # plt.plot(n_wide_beams,learnable_codebook_topk_acc[:,2],'--',marker='s',label='[10], k={}'.format(3))
        # plt.plot(n_wide_beams,learnable_codebook_topk_acc[:,1],'-.',marker='s',label='[10], k={}'.format(2))
        # plt.plot(n_wide_beams,learnable_codebook_topk_acc[:,0],marker='s',label='[10], k={}'.format(1))
        # plt.hlines(y=exhaustive_acc,xmin=min(n_wide_beams),xmax=max(n_wide_beams),linestyles='dotted',label='Exhaustive search')
        # plt.xticks(n_wide_beams)
        # plt.legend()
        # plt.xlabel('probing codebook size')
        # plt.ylabel('Accuracy')
        # plt.title('Optimal narrow beam prediction accuracy_cbnoise_{}'.format(cb_noise))
        # plt.show()
        


        # # plt.figure(figsize=(6,4.5))
        # # plt.plot(n_wide_beams,dft_codebook_topk_acc[:,2],'--',marker='s',label='DFT, k={}'.format(3))
        # # plt.plot(n_wide_beams,dft_codebook_topk_acc[:,1],'-.',marker='s',label='DFT, k={}'.format(2))
        # # plt.plot(n_wide_beams,dft_codebook_topk_acc[:,0],marker='s',label='DFT, k={}'.format(1))
        # # plt.hlines(y=exhaustive_acc,xmin=min(n_wide_beams),xmax=max(n_wide_beams),linestyles='dotted',label='Exhaustive search')
        # # plt.xticks(n_wide_beams)
        # # plt.legend()
        # # plt.xlabel('probing codebook size')
        # # plt.ylabel('Accuracy')
        # # plt.title('Optimal narrow beam prediction accuracy')
        # # plt.show()

        # plt.figure(figsize=(6,4.5))
        # plt.plot(n_wide_beams,proposed_codebook_topk_acc[:,2],'--',marker='s',label='Proposed method, k={}'.format(3))
        # plt.plot(n_wide_beams,proposed_codebook_topk_acc[:,1],'-.',marker='s',label='Proposed method, k={}'.format(2))
        # plt.plot(n_wide_beams,proposed_codebook_topk_acc[:,0],marker='s',label='Proposed method, k={}'.format(1))
        # plt.hlines(y=exhaustive_acc,xmin=min(n_wide_beams),xmax=max(n_wide_beams),linestyles='dotted',label='Exhaustive search')
        # plt.xticks(n_wide_beams)
        # plt.legend()
        # plt.xlabel('probing codebook size')
        # plt.ylabel('Accuracy')
        # plt.title('Optimal narrow beam prediction accuracy_cbnoise_{}'.format(cb_noise))
        # plt.show()

        # "Average SNRs"
        # plt.figure(figsize=(6,4.5))
        # plt.plot(n_wide_beams,learned_codebook_topk_snr[:,:,2].mean(axis=1),'--',marker='s',label='[10], k={}'.format(3))
        # plt.plot(n_wide_beams,learned_codebook_topk_snr[:,:,1].mean(axis=1),'-.',marker='s',label='[10], k={}'.format(2))
        # plt.plot(n_wide_beams,learned_codebook_topk_snr[:,:,0].mean(axis=1),marker='s',label='[10], k={}'.format(1))
        # plt.xticks(n_wide_beams)
        # plt.legend()
        # plt.xlabel('probing codebook size')
        # plt.ylabel('Average SNR (dB)')
        # plt.title('SNR of top-k predicted beams_cbnoise_{}'.format(cb_noise))
        # plt.show()

        # # plt.figure(figsize=(6,4.5))
        # # plt.plot(n_wide_beams,dft_codebook_topk_snr[:,:,2].mean(axis=1),'--',marker='s',label='DFT, k={}'.format(3))
        # # plt.plot(n_wide_beams,dft_codebook_topk_snr[:,:,1].mean(axis=1),'-.',marker='s',label='DFT, k={}'.format(2))
        # # plt.plot(n_wide_beams,dft_codebook_topk_snr[:,:,0].mean(axis=1),marker='s',label='DFT, k={}'.format(1))
        # # plt.xticks(n_wide_beams)
        # # plt.legend()
        # # plt.xlabel('probing codebook size')
        # # plt.ylabel('Average SNR (dB)')
        # # plt.title('SNR of top-k predicted beams')
        # # plt.show()

        # plt.figure(figsize=(6,4.5))
        # plt.plot(n_wide_beams,proposed_codebook_topk_snr[:,:,2].mean(axis=1),'--',marker='s',label='Proposed, k={}'.format(3))
        # plt.plot(n_wide_beams,proposed_codebook_topk_snr[:,:,1].mean(axis=1),'-.',marker='s',label='Proposed, k={}'.format(2))
        # plt.plot(n_wide_beams,proposed_codebook_topk_snr[:,:,0].mean(axis=1),marker='s',label='Proposed, k={}'.format(1))
        # plt.xticks(n_wide_beams)
        # plt.legend()
        # plt.xlabel('probing codebook size')
        # plt.ylabel('Average SNR (dB)')
        # plt.title('SNR of top-k predicted beams_cbnoise_{}'.format(cb_noise))
        # plt.show()

        # # for i,N in enumerate(n_wide_beams):     
        # #     fig,ax = plot_codebook_pattern(learned_codebooks[i].T)
        # #     ax.set_title('Trainable {}-Beam Codebook'.format(N))
        # #     fig,ax = plot_codebook_pattern(dft_codebooks[i])
        # #     ax.set_title('DFT {}-Beam Codebook'.format(N))
        # #     fig,ax = plot_codebook_pattern(AMCF_codebooks[i])
        # #     ax.set_title('AMCF {}-Beam Codebook'.format(N))

# %%
plt.figure(figsize=(6,4.5))
plotpt=len(cb_noise_list)
plt.plot(cb_noise_list[0:plotpt],learned_snr_cb_noise_list[2,0:plotpt],'--',marker='s',label='Ethan, k={}'.format(3))
plt.plot(cb_noise_list[0:plotpt],learned_snr_cb_noise_list[1,0:plotpt],'--',marker='s',label='Ethan, k={}'.format(2))
plt.plot(cb_noise_list[0:plotpt],learned_snr_cb_noise_list[0,0:plotpt],'--',marker='s',label='Ethan, k={}'.format(1))
plt.plot(cb_noise_list[0:plotpt],proposed_snr_cb_noise_list[2,0:plotpt],'-',marker='o',label='Proposed, k={}'.format(3))
plt.plot(cb_noise_list[0:plotpt],proposed_snr_cb_noise_list[1,0:plotpt],'-',marker='o',label='Proposed, k={}'.format(2))
plt.plot(cb_noise_list[0:plotpt],proposed_snr_cb_noise_list[0,0:plotpt],'-',marker='o',label='Proposed, k={}'.format(1))

plt.xticks(cb_noise_list[0:plotpt])
plt.legend()
plt.xlabel('cb_noise sigma')
plt.ylabel('Average SNR (dB)')
plt.title('SNR of top-k predicted beams')
plt.show()


plt.figure(figsize=(6,4.5))
plotpt=6#len(cb_noise_list)
plt.plot(cb_noise_list[0:plotpt],learned_acc_cb_noise_list[2,0:plotpt],'--',marker='s',label='Ethan, k={}'.format(3))
plt.plot(cb_noise_list[0:plotpt],learned_acc_cb_noise_list[1,0:plotpt],'--',marker='s',label='Ethan, k={}'.format(2))
plt.plot(cb_noise_list[0:plotpt],learned_acc_cb_noise_list[0,0:plotpt],'--',marker='s',label='Ethan, k={}'.format(1))
plt.plot(cb_noise_list[0:plotpt],proposed_acc_cb_noise_list[2,0:plotpt],'-',marker='o',label='Proposed, k={}'.format(3))
plt.plot(cb_noise_list[0:plotpt],proposed_acc_cb_noise_list[1,0:plotpt],'-',marker='o',label='Proposed, k={}'.format(2))
plt.plot(cb_noise_list[0:plotpt],proposed_acc_cb_noise_list[0,0:plotpt],'-',marker='o',label='Proposed, k={}'.format(1))

plt.xticks(cb_noise_list[0:plotpt])
plt.legend()
plt.xlabel('cb_noise sigma')
plt.ylabel('Accuracy')
plt.title('Top-k beam prediction accuracy')
plt.show()

