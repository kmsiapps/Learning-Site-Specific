# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 12:49:09 2021

@author: Yuqiang (Ethan) Heng
"""
import torch
import numpy as np
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn import Module
import torch.nn as nn
from beam_utils_L import codebook_blockmatrix, codebook_blockmatrix2

    
class PhaseShifter(Module):
    """
    This module is a pytorch implementation of the complex Dense layer for phase shifters satisfying constant modulus constraint
    It is insipred by the implemention by the orignal authors of the Deep Complex Networks paper
    https://github.com/ChihebTrabelsi/deep_complex_networks
    
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        scale: scaling factor to account for tx and noise power
        theta: the initial phase shift values
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    scale: float
    theta: Tensor

    def __init__(self, in_features: int, out_features: int, scale: float=1, theta = None) -> None:
        super(PhaseShifter, self).__init__()
        self.in_features = in_features
        self.in_dim = self.in_features//2
        self.out_features = out_features
        self.scale = scale
        self.theta = Parameter(torch.Tensor(self.in_dim, self.out_features)) 
        self.reset_parameters(theta)

    def reset_parameters(self, theta = None) -> None:
        if theta is None:
            init.uniform_(self.theta, a=0, b=2*np.pi)
        else:
            assert theta.shape == (self.in_dim,self.out_features)
            self.theta = Parameter(theta) 
        self.real_kernel = (1 / self.scale) * torch.cos(self.theta)  #
        self.imag_kernel = (1 / self.scale) * torch.sin(self.theta)  #
    
    def forward(self, inputs: Tensor) -> Tensor:
        self.real_kernel = (1 / self.scale) * torch.cos(self.theta)  #
        self.imag_kernel = (1 / self.scale) * torch.sin(self.theta)  #        
        cat_kernels_4_real = torch.cat(
            (self.real_kernel, -self.imag_kernel),
            dim=-1
        )
        cat_kernels_4_imag = torch.cat(
            (self.imag_kernel, self.real_kernel),
            dim=-1
        )
        cat_kernels_4_complex = torch.cat(
            (cat_kernels_4_real, cat_kernels_4_imag),
            dim=0
        )  # This block matrix represents the conjugate transpose of the original:
        # [ W_R, -W_I; W_I, W_R]

        output = torch.matmul(inputs, cat_kernels_4_complex)
        return output

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )
    
    def get_theta(self) -> torch.Tensor:
        return self.theta.detach().clone()
    
    def get_weights(self) -> torch.Tensor:
        with torch.no_grad():
            real_kernel = (1 / self.scale) * torch.cos(self.theta)  #
            imag_kernel = (1 / self.scale) * torch.sin(self.theta)  #        
            beam_weights = real_kernel + 1j*imag_kernel
        return beam_weights

def conv_dimension(input_dim,kernel_size,padding=0,dilation=1,stride=1):
    return np.floor((input_dim+2*padding-dilation*(kernel_size-1)-1)/stride+1)
    

class ComputePower(Module):
    def __init__(self, in_shape):
        super(ComputePower, self).__init__()
        self.shape = in_shape
        self.len_real = int(self.shape/2)

    def forward(self, x):
        real_part = x[:,:self.len_real]
        imag_part = x[:,self.len_real:]
        sq_real = torch.pow(real_part,2)
        sq_imag = torch.pow(imag_part,2)
        # abs_values = torch.pow(sq_real + sq_imag,0.5)
        abs_values = sq_real + sq_imag
        return abs_values
    
class Beam_Classifier(nn.Module):
    def __init__(self, n_antenna, n_wide_beam, n_narrow_beam, trainable_codebook = True, theta = None, complex_codebook=None, noise_power = 0.0, norm_factor = 1.0):
        super(Beam_Classifier, self).__init__()
        self.trainable_codebook = trainable_codebook
        self.n_antenna = n_antenna
        self.n_wide_beam = n_wide_beam
        self.n_narrow_beam = n_narrow_beam
        self.noise_power = float(noise_power)
        self.norm_factor = float(norm_factor)
        if trainable_codebook:
            self.codebook = PhaseShifter(in_features=2*n_antenna, out_features=n_wide_beam, scale=np.sqrt(n_antenna), theta=theta)
            self.initial_codebook = self.codebook.get_weights().detach().clone().numpy()
        else:            
            self.complex_codebook = complex_codebook # n_beams x n_antenna
            cb_blockmatrix = codebook_blockmatrix(self.complex_codebook.T)
            self.codebook = torch.from_numpy(cb_blockmatrix).float()
            self.codebook.requires_grad = False
            
        self.compute_power = ComputePower(2*n_wide_beam)
        self.relu = nn.ReLU()
        self.dense1 = nn.Linear(in_features=n_wide_beam, out_features=2*n_wide_beam)
        self.dense2 = nn.Linear(in_features=2*n_wide_beam, out_features=3*n_wide_beam)
        self.dense3 = nn.Linear(in_features=3*n_wide_beam, out_features=n_narrow_beam)
        self.softmax = nn.Softmax()
    
    def forward(self, x):
        if self.trainable_codebook:
            bf_signal = self.codebook(x)
        else:
            bf_signal = torch.matmul(x,self.codebook)
        noise_vec = torch.normal(0,1, size=bf_signal.size(), device=bf_signal.device)*torch.sqrt(torch.tensor([self.noise_power/2], device=bf_signal.device))/torch.tensor([self.norm_factor], device=bf_signal.device)
        bf_signal = bf_signal + noise_vec
        bf_power = self.compute_power(bf_signal)
        out = self.relu(bf_power)
        out = self.relu(self.dense1(out))
        out = self.relu(self.dense2(out))
        out = self.dense3(out)
        return out
    def get_codebook(self) -> np.ndarray:
        if self.trainable_codebook:
            return self.codebook.get_weights().detach().clone().cpu().numpy()
        else:
            return self.complex_codebook

class Beam_Classifier_Prune(nn.Module):
    def __init__(self, n_antenna, n_wide_beam, n_narrow_beam, trainable_codebook = True, theta = None, complex_codebook=None, noise_power = 0.0, norm_factor = 1.0, prune_idx = None):
        super(Beam_Classifier_Prune, self).__init__()
        self.trainable_codebook = trainable_codebook
        self.n_antenna = n_antenna
        self.n_wide_beam = n_wide_beam
        self.n_narrow_beam = n_narrow_beam
        self.noise_power = float(noise_power)
        self.norm_factor = float(norm_factor)
        self.prune_idx = np.array(prune_idx, dtype=np.int64) if prune_idx is not None else np.array([], dtype=np.int64)

        if trainable_codebook:
            self.codebook = PhaseShifter(in_features=2*n_antenna, out_features=n_wide_beam, scale=np.sqrt(n_antenna), theta=theta)
            self.initial_codebook = self.codebook.get_weights().detach().clone().numpy()
        else:            
            self.complex_codebook = complex_codebook # n_beams x n_antenna
            cb_blockmatrix = codebook_blockmatrix(self.complex_codebook.T)
            self.codebook = torch.from_numpy(cb_blockmatrix).float()
            self.codebook.requires_grad = False
            
        self.compute_power = ComputePower(2*n_wide_beam)
        self.relu = nn.ReLU()
        self.dense1 = nn.Linear(in_features=n_wide_beam, out_features=2*n_wide_beam)
        self.dense2 = nn.Linear(in_features=2*n_wide_beam, out_features=3*n_wide_beam)
        self.dense3 = nn.Linear(in_features=3*n_wide_beam, out_features=n_narrow_beam)
        self.softmax = nn.Softmax()
    
    def forward(self, x): 
        if self.trainable_codebook:
            bf_signal = self.codebook(x)
        else:
            bf_signal = torch.matmul(x,self.codebook)
        
        if self.prune_idx is not None and len(self.prune_idx) > 0:
            bf_signal[:, self.prune_idx] = 0. # real idx
            bf_signal[:, bf_signal.shape[-1]//2 + self.prune_idx] = 0. # imag idx

        noise_vec = torch.normal(0,1, size=bf_signal.size(), device=bf_signal.device)*torch.sqrt(torch.tensor([self.noise_power/2], device=bf_signal.device))/torch.tensor([self.norm_factor], device=bf_signal.device)
        bf_signal = bf_signal + noise_vec
        bf_power = self.compute_power(bf_signal)
        out = self.relu(bf_power)
        out = self.relu(self.dense1(out))
        out = self.relu(self.dense2(out))
        out = self.dense3(out)
        return out
    def get_codebook(self) -> np.ndarray:
        if self.trainable_codebook:
            return self.codebook.get_weights().detach().clone().cpu().numpy()
        else:
            return self.complex_codebook
        

class Beam_Classifier_Quantization(nn.Module):
    def __init__(self, dB_stepsize, n_antenna, n_wide_beam, n_narrow_beam, trainable_codebook = True, theta = None, complex_codebook=None, noise_power = 0.0, norm_factor = 1.0):
        super(Beam_Classifier_Quantization, self).__init__()
        self.trainable_codebook = trainable_codebook
        self.n_antenna = n_antenna
        self.n_wide_beam = n_wide_beam
        self.n_narrow_beam = n_narrow_beam
        self.noise_power = float(noise_power)
        self.norm_factor = float(norm_factor)
        if trainable_codebook:
            self.codebook = PhaseShifter(in_features=2*n_antenna, out_features=n_wide_beam, scale=np.sqrt(n_antenna), theta=theta)
            self.initial_codebook = self.codebook.get_weights().detach().clone().numpy()
        else:            
            self.complex_codebook = complex_codebook # n_beams x n_antenna
            cb_blockmatrix = codebook_blockmatrix(self.complex_codebook.T)
            self.codebook = torch.from_numpy(cb_blockmatrix).float()
            self.codebook.requires_grad = False
            
        self.compute_power = ComputePower(2*n_wide_beam)
        self.dB_stepsize = dB_stepsize
        self.relu = nn.ReLU()
        self.dense1 = nn.Linear(in_features=n_wide_beam, out_features=2*n_wide_beam)
        self.dense2 = nn.Linear(in_features=2*n_wide_beam, out_features=3*n_wide_beam)
        self.dense3 = nn.Linear(in_features=3*n_wide_beam, out_features=n_narrow_beam)
        self.softmax = nn.Softmax()
    
    def forward(self, x):
        if self.trainable_codebook:
            bf_signal = self.codebook(x)
        else:
            bf_signal = torch.matmul(x,self.codebook)
        noise_vec = torch.normal(0,1, size=bf_signal.size(), device=bf_signal.device)*torch.sqrt(torch.tensor([self.noise_power/2], device=bf_signal.device))/torch.tensor([self.norm_factor], device=bf_signal.device)
        bf_signal = bf_signal + noise_vec
        bf_power = self.compute_power(bf_signal)
        if self.dB_stepsize==0:
            quantized_bf_power = bf_power
        else:
            bf_power_dB = 10*torch.log10(bf_power)
            quantized_bf_power_dB = torch.round(bf_power_dB/self.dB_stepsize)*self.dB_stepsize
            quantized_bf_power =10**(quantized_bf_power_dB/10)
        out = self.relu(quantized_bf_power)
        out = self.relu(self.dense1(out))
        out = self.relu(self.dense2(out))
        out = self.dense3(out)
        return out
    def get_codebook(self) -> np.ndarray:
        if self.trainable_codebook:
            return self.codebook.get_weights().detach().clone().cpu().numpy()
        else:
            return self.complex_codebook

        
def fit(model,top_k, train_loader, val_loader, opt, loss_fn, EPOCHS):
    optimizer = opt
    device = next(model.parameters()).device
    if hasattr(model, 'codebook') and isinstance(model.codebook, torch.Tensor):
        model.codebook = model.codebook.to(device)
    train_loss_hist = []
    val_loss_hist = []
    train_acc_hist=[]
    val_acc_hist=[]
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        train_acc = 0
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            var_X_batch = X_batch.float().to(device, non_blocking=True)
            var_y_batch = y_batch.long().to(device, non_blocking=True)
            optimizer.zero_grad()
            output = model(var_X_batch)
            loss = loss_fn(output, var_y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().item()
            train_acc += (output.argmax(dim=1) == var_y_batch).sum().item()/var_y_batch.shape[0]
        train_loss /= batch_idx + 1
        train_acc /= batch_idx + 1
        model.eval()
        val_loss = 0
        val_acc = 0
        for batch_idx, (X_batch, y_batch) in enumerate(val_loader):
            var_X_batch = X_batch.float().to(device, non_blocking=True)
            var_y_batch = y_batch.long().to(device, non_blocking=True)  
            output = model(var_X_batch)
            loss = loss_fn(output, var_y_batch)
            val_loss += loss.detach().item()
            # val_acc += (output.argmax(dim=1) == var_y_batch).sum().item()/var_y_batch.shape[0]
            _,topkbeamind=torch.topk(output,top_k,dim=1)
            val_acc += torch.sum(torch.any(topkbeamind == var_y_batch.unsqueeze(1),dim=1)).item()/var_y_batch.shape[0]
        val_loss /= batch_idx + 1
        val_acc /= batch_idx + 1
        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        train_acc_hist.append(train_acc)
        val_acc_hist.append(val_acc)
        if epoch % 10 == 0:
            print('Epoch : {}, Top_{}, Training loss = {:.2f}, Training Acc = {:.2f}, Validation loss = {:.2f}, Validation Acc = {:.2f}.'.format(epoch,top_k,train_loss,train_acc,val_loss,val_acc))
    return train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist

def eval_model(model, test_loader, loss_fn):
    device = next(model.parameters()).device
    if hasattr(model, 'codebook') and isinstance(model.codebook, torch.Tensor):
        model.codebook = model.codebook.to(device)
    model.eval()
    test_loss = 0
    test_acc = 0
    for batch_idx, (X_batch, y_batch) in enumerate(test_loader):
        var_X_batch = X_batch.float().to(device, non_blocking=True)
        var_y_batch = y_batch.long().to(device, non_blocking=True)  
        output = model(var_X_batch)
        loss = loss_fn(output, var_y_batch)
        test_loss += loss.detach().item()
        test_acc += (output.argmax(dim=1) == var_y_batch).sum().item()/var_y_batch.shape[0]
    test_loss /= batch_idx + 1
    test_acc /= batch_idx + 1
    return test_loss, test_acc

class projection_trainer(nn.Module):
    def __init__(self, n_wide_beam, n_narrow_beam, narrow_beam_CB, noise_power = 0.0, norm_factor = 1.0):
        super(projection_trainer, self).__init__()
        self.n_wide_beam = n_wide_beam
        self.n_narrow_beam = n_narrow_beam
        self.noise_power = float(noise_power)
        self.norm_factor = float(norm_factor)
        self.narrow_beam_CB = torch.as_tensor(narrow_beam_CB,dtype=torch.complex64)
        self.compute_power = ComputePower(2*n_wide_beam)
        self.softmax = nn.Softmax()
        self.projector = Projection(n_narrow_beam,n_wide_beam)
        self.relu = nn.ReLU()
        self.dense1 = nn.Linear(in_features=n_wide_beam, out_features=2*n_wide_beam)
        self.dense2 = nn.Linear(in_features=2*n_wide_beam, out_features=3*n_wide_beam)
        self.dense3 = nn.Linear(in_features=3*n_wide_beam, out_features=n_narrow_beam)
    def forward(self, x):
        # out = self.projector(x)
        # out = out.unsqueeze(1)
        # out = self.CNN_Classifier(out)
        out = self.projector(x)
        out = self.compute_power(out)
        out = self.relu(out)
        out = self.relu(self.dense1(out))
        out = self.relu(self.dense2(out))
        out = self.dense3(out)
        out = out.squeeze()
        return out
    
    def get_projection_L(self) -> np.ndarray:
        return self.projector.get_weights().clone().detach().numpy()
    
    def get_initial_L(self) -> np.ndarray:
        return self.projector.get_init_weights().clone().detach().numpy()
    
    def get_regul_term(self):
        temp = self.projector.get_weights().clone().detach()
        temp = temp.T 
        #a=temp[:temp.shape[0]//2,:temp.shape[1]//2]+1j*temp[temp.shape[0]//2:,temp.shape[1]//2:]
        # print(a.dtype)
        W=torch.matmul(temp[:temp.shape[0]//2,:temp.shape[1]//2]+1j*temp[temp.shape[0]//2:,temp.shape[1]//2:], self.narrow_beam_CB)
        regul_term = torch.norm(torch.abs(W) - torch.ones(W.shape[0],W.shape[1]))
        return regul_term
    
    

    
class Projection(nn.Module):
    def __init__(self, n_nb: int, n_wb: int):
        super(Projection, self).__init__()
        self.n_nb = n_nb
        self.n_wb = n_wb
        self.projection_L_complex= nn.Parameter(torch.complex(torch.randn(n_wb,n_nb),torch.randn(n_wb,n_nb)))
        projection_L_block_matrix = codebook_blockmatrix2(self.projection_L_complex.T)
        self.projection_L = nn.Parameter(torch.from_numpy(projection_L_block_matrix).float().requires_grad_(True))
        self.initial_L = self.projection_L.clone().detach()

    def forward(self, inputs: Tensor) -> Tensor:
        output = torch.matmul(inputs, self.projection_L)
        return output
    
    def get_weights(self) -> np.ndarray:
        return self.projection_L.clone().detach().cpu()
    def get_init_weights(self) -> np.ndarray:
        return self.initial_L.clone().detach().cpu()

def fit_projector(model,top_k, narrow_beam_CB, narrow_beam_CB_inv, train_loader, val_loader, opt, loss_fn, EPOCHS):
    optimizer = opt
    train_loss_hist = []
    val_loss_hist = []
    train_acc_hist=[]
    val_acc_hist=[]
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        train_acc = 0
            
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            var_X_batch = X_batch.float()
            var_y_batch = y_batch.long()
            optimizer.zero_grad()
            output = model(var_X_batch)
            # regul_term = model.get_regul_term()
            loss =loss_fn(output, var_y_batch) 
            loss.backward()
            optimizer.step()

            if batch_idx%10==0:
                with torch.no_grad():
                    a = custom_projection_L(model.projector.projection_L,narrow_beam_CB,narrow_beam_CB_inv)
                    model.projector.projection_L.data = a
            
                    
           
            train_loss += loss.detach().item()
            train_acc += (output.argmax(dim=1) == var_y_batch).sum().item()/var_y_batch.shape[0]
        train_loss /= batch_idx + 1
        train_acc /= batch_idx + 1

        # if epoch % 10 == 0:
        #     model.projector.projection_L.data = custom_projection_L(model.projector.projection_L,narrow_beam_CB,narrow_beam_CB_inv)

        model.eval()
        with torch.no_grad():
            val_loss = 0
            val_acc = 0
            for batch_idx, (X_batch, y_batch) in enumerate(val_loader):
                var_X_batch = X_batch.float()
                var_y_batch = y_batch.long()  
                output = model(var_X_batch)
                # regul_term = model.get_regul_term()
                loss = loss_fn(output, var_y_batch)
                val_loss += loss.detach().item()
                # val_acc += (output.argmax(dim=1) == var_y_batch).sum().itsem()/var_y_batch.shape[0]
                _,topkbeamind=torch.topk(output,top_k,dim=1)
                val_acc += torch.sum(torch.any(topkbeamind == var_y_batch.unsqueeze(1),dim=1)).item()/var_y_batch.shape[0]
            val_loss /= batch_idx + 1
            val_acc /= batch_idx + 1
            train_loss_hist.append(train_loss)
            val_loss_hist.append(val_loss)
            train_acc_hist.append(train_acc)
            val_acc_hist.append(val_acc)
        if epoch == EPOCHS-1:
            a = custom_projection_L(model.projector.projection_L,narrow_beam_CB,narrow_beam_CB_inv)
            model.projector.projection_L.data = a
        if epoch % 10 == 0:
            print('Epoch : {}, Top_{}, Training loss = {:.2f}, Training Acc = {:.2f}, Validation loss = {:.2f}, Validation Acc = {:.2f}.'.format(epoch,top_k,train_loss,train_acc,val_loss,val_acc))

                
    return train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist

def custom_projection_L(weights: nn.Parameter,narrow_beam_CB,narrow_beam_CB_inv):
    weights_complex=weights[:weights.shape[0]//2,:weights.shape[1]//2]+1j*weights[:weights.shape[0]//2,weights.shape[1]//2:]
    weights_complex = weights_complex.conj().T
    weights_complex_np = weights_complex.detach().clone().numpy()
    V= narrow_beam_CB
    w=torch.matmul(weights_complex,V)
    w_normalized = w/(torch.sqrt(torch.as_tensor(V.shape[1],dtype=torch.float32))*torch.abs(w))
    
    if V.shape[1]>=V.shape[0]:
        modified_weights_complex = torch.transpose(torch.conj(torch.matmul(w_normalized, narrow_beam_CB_inv)),0,1)
    else:
        w_normalized_np = w_normalized.detach().clone().numpy()
        V_np = V.detach().clone().numpy()
        modified_weights_complex_np = np.linalg.lstsq(V_np.T,w_normalized_np.T)[0].T
        null_proj = (np.identity(V_np.shape[0]) - V_np.conj()@np.linalg.inv(V_np.T@V_np.conj())@V_np.T)@weights_complex_np.T
        modified_weights_complex_np = (modified_weights_complex_np + null_proj.T).conj().T
        # modified_weights_complex_np = (modified_weights_complex_np ).conj().T
        modified_weights_complex = torch.as_tensor(modified_weights_complex_np,dtype=torch.complex64)
        # print(torch.abs(torch.matmul(modified_weights_complex,narrow_beam_CB)))
    modified_weights = codebook_blockmatrix2(modified_weights_complex)
    modified_weights = torch.as_tensor(modified_weights,dtype=torch.float32)
    return modified_weights


class projection_trainer_fake(nn.Module):
    def __init__(self, if_fake,n_wide_beam, n_narrow_beam, narrow_beam_CB, noise_power = 0.0, norm_factor = 1.0):
        super(projection_trainer_fake, self).__init__()
        self.if_fake = if_fake
        self.n_wide_beam = n_wide_beam
        self.n_narrow_beam = n_narrow_beam
        self.noise_power = float(noise_power)
        self.norm_factor = float(norm_factor)
        self.narrow_beam_CB = torch.as_tensor(narrow_beam_CB,dtype=torch.complex64)
        self.compute_power = ComputePower(2*n_wide_beam)
        self.softmax = nn.Softmax()
        self.projector = Projection(n_narrow_beam,n_wide_beam)
        self.relu = nn.ReLU()
        self.dense1 = nn.Linear(in_features=n_wide_beam, out_features=2*n_wide_beam)
        self.dense2 = nn.Linear(in_features=2*n_wide_beam, out_features=3*n_wide_beam)
        self.dense3 = nn.Linear(in_features=3*n_wide_beam, out_features=n_narrow_beam)
        
    def forward(self, x):
        # rand_theta = np.random.uniform(-np.pi,np.pi,(x.shape[0],x.shape[1]//2))
        # fake_x_real = np.multiply(x[:,0:x.shape[1]//2],np.cos(rand_theta)) - np.multiply(x[:,x.shape[1]//2:],np.sin(rand_theta))
        # fake_x_imag = np.multiply(x[:,0:x.shape[1]//2],np.sin(rand_theta)) + np.multiply(x[:,x.shape[1]//2:],np.cos(rand_theta))
        # fake_x =torch.as_tensor(np.concatenate((fake_x_real,fake_x_imag),axis=1),dtype=torch.float32)
        if self.if_fake:
            rand_theta = torch.rand_like(x) * (2 * np.pi) - np.pi
            fake_x_real = x * torch.cos(rand_theta)
            fake_x_imag = x * torch.sin(rand_theta)
            fake_x = torch.cat((fake_x_real, fake_x_imag), dim=1).float()
        else:
            fake_x = x
        out = self.projector(fake_x)
        out = self.compute_power(out)
        out = self.relu(out)
        out = self.relu(self.dense1(out))
        out = self.relu(self.dense2(out))
        out = self.dense3(out)
        out = out.squeeze()
        return out
    
    def get_projection_L(self) -> np.ndarray:
        return self.projector.get_weights().clone().detach().numpy()
    
    def get_initial_L(self) -> np.ndarray:
        return self.projector.get_init_weights().clone().detach().numpy()
    
    def get_regul_term(self):
        temp = self.projector.get_weights().clone().detach()
        temp = temp.T 
        #a=temp[:temp.shape[0]//2,:temp.shape[1]//2]+1j*temp[temp.shape[0]//2:,temp.shape[1]//2:]
        # print(a.dtype)
        W=torch.matmul(temp[:temp.shape[0]//2,:temp.shape[1]//2]+1j*temp[temp.shape[0]//2:,temp.shape[1]//2:], self.narrow_beam_CB)
        regul_term = torch.norm(torch.abs(W) - torch.ones(W.shape[0],W.shape[1]))
        return regul_term




class projection_trainer_classifieronly(nn.Module):
    def __init__(self, if_fake,n_wide_beam, n_narrow_beam, narrow_beam_CB, noise_power = 0.0, norm_factor = 1.0):
        super(projection_trainer_classifieronly, self).__init__()
        self.if_fake = if_fake
        self.n_wide_beam = n_wide_beam
        self.n_narrow_beam = n_narrow_beam
        self.noise_power = float(noise_power)
        self.norm_factor = float(norm_factor)
        self.narrow_beam_CB = torch.as_tensor(narrow_beam_CB,dtype=torch.complex64)
        self.compute_power = ComputePower(2*n_wide_beam)
        self.softmax = nn.Softmax()
        self.projector = Projection(n_narrow_beam,n_wide_beam)
        self.relu = nn.ReLU()
        self.dense1 = nn.Linear(in_features=n_wide_beam, out_features=2*n_wide_beam)
        self.dense2 = nn.Linear(in_features=2*n_wide_beam, out_features=3*n_wide_beam)
        self.dense3 = nn.Linear(in_features=3*n_wide_beam, out_features=n_narrow_beam)
        
    def forward(self, x):
        # rand_theta = np.random.uniform(-np.pi,np.pi,(x.shape[0],x.shape[1]//2))
        # fake_x_real = np.multiply(x[:,0:x.shape[1]//2],np.cos(rand_theta)) - np.multiply(x[:,x.shape[1]//2:],np.sin(rand_theta))
        # fake_x_imag = np.multiply(x[:,0:x.shape[1]//2],np.sin(rand_theta)) + np.multiply(x[:,x.shape[1]//2:],np.cos(rand_theta))
        # fake_x =torch.as_tensor(np.concatenate((fake_x_real,fake_x_imag),axis=1),dtype=torch.float32)

        
        out = x # assume probing beam measurements.
        out = self.relu(out)
        out = self.relu(self.dense1(out))
        out = self.relu(self.dense2(out))
        out = self.dense3(out)
        out = out.squeeze()
        return out
    
    def get_projection_L(self) -> np.ndarray:
        return self.projector.get_weights().clone().detach().numpy()
    
    def get_initial_L(self) -> np.ndarray:
        return self.projector.get_init_weights().clone().detach().numpy()
    
    def get_regul_term(self):
        temp = self.projector.get_weights().clone().detach()
        temp = temp.T 
        #a=temp[:temp.shape[0]//2,:temp.shape[1]//2]+1j*temp[temp.shape[0]//2:,temp.shape[1]//2:]
        # print(a.dtype)
        W=torch.matmul(temp[:temp.shape[0]//2,:temp.shape[1]//2]+1j*temp[temp.shape[0]//2:,temp.shape[1]//2:], self.narrow_beam_CB)
        regul_term = torch.norm(torch.abs(W) - torch.ones(W.shape[0],W.shape[1]))
        return regul_term





# class projection_trainer_RSRP(nn.Module):
#     def __init__(self, n_wide_beam, n_narrow_beam, narrow_beam_CB, noise_power = 0.0, norm_factor = 1.0):
#         super(projection_trainer_RSRP, self).__init__()
#         self.n_wide_beam = n_wide_beam
#         self.n_narrow_beam = n_narrow_beam
#         self.noise_power = float(noise_power)
#         self.norm_factor = float(norm_factor)
#         self.narrow_beam_CB = torch.as_tensor(narrow_beam_CB,dtype=torch.complex64)
#         self.compute_power = ComputePower(2*n_narrow_beam)
#         self.compute_power2 = ComputePower(2*n_wide_beam)
#         self.CNN_Classifier = CNN_Classifier(n_wide_beam, n_narrow_beam)
#         self.projector = Projection_RSRP(n_narrow_beam,n_wide_beam)
#         self.relu = nn.ReLU()
#         self.dense1 = nn.Linear(in_features=n_wide_beam, out_features=2*n_wide_beam)
#         self.dense2 = nn.Linear(in_features=2*n_wide_beam, out_features=3*n_wide_beam)
#         self.dense3 = nn.Linear(in_features=3*n_wide_beam, out_features=n_narrow_beam)
#         self.softmax = nn.Softmax()
#     def forward(self, x):
#         nb_measurement_RSRP = self.compute_power(x)
#         out = self.projector(nb_measurement_RSRP)
#         out = torch.pow(torch.abs(out),1)
#         out = self.relu(out)
#         out = self.relu(self.dense1(out))
#         out = self.relu(self.dense2(out))
#         out = self.dense3(out)
#         out = out.squeeze()
#         return out
    
#     def get_projection_L(self) -> np.ndarray:
#         return self.projector.get_weights().clone().detach().numpy()
    
#     def get_initial_L(self) -> np.ndarray:
#         return self.projector.get_init_weights().clone().detach().numpy()
    
#     def get_regul_term(self):
#         temp = self.projector.get_weights().clone().detach()
#         temp = temp.T 
#         W=torch.matmul(temp[:temp.shape[0]//2,:temp.shape[1]//2]+1j*temp[temp.shape[0]//2:,temp.shape[1]//2:], self.narrow_beam_CB)
#         regul_term = torch.norm(torch.abs(W) - torch.ones(W.shape[0],W.shape[1]))
#         return regul_term
    

# class Projection_RSRP(nn.Module):
#     def __init__(self, n_nb: int, n_wb: int):
#         super(Projection_RSRP, self).__init__()
#         self.n_nb = n_nb
#         self.n_wb = n_wb
#         self.projection_L= nn.Parameter(torch.randn(n_wb,n_nb)).T
#         self.initial_L = self.projection_L.clone().detach()

#     def forward(self, inputs: Tensor) -> Tensor:
#         output = torch.matmul(inputs, self.projection_L)
#         return output
    
#     def get_weights(self) -> np.ndarray:
#         return self.projection_L.clone().detach()
#     def get_init_weights(self) -> np.ndarray:
#         return self.initial_L.clone().detach()
    
# def custom_projection_L_RSRP(weights: nn.Parameter,narrow_beam_CB,narrow_beam_CB_inv):
#     weights=weights.to(torch.complex64)
#     w=torch.matmul(weights.T,narrow_beam_CB)
#     w_normalized = w/torch.abs(w)
#     modified_weights = torch.matmul(w_normalized, narrow_beam_CB_inv)   
#     modified_weights = torch.as_tensor(modified_weights.T,dtype=torch.float32)                            
#     return modified_weights

# def fit_projector_RSRP(model,top_k, narrow_beam_CB, narrow_beam_CB_inv, train_loader, val_loader, opt, loss_fn, EPOCHS):
#     optimizer = opt
#     train_loss_hist = []
#     val_loss_hist = []
#     train_acc_hist=[]
#     val_acc_hist=[]
#     for epoch in range(EPOCHS):
#         model.train()
#         train_loss = 0
#         train_acc = 0
#         for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
#             var_X_batch = X_batch.float()
#             var_y_batch = y_batch.long()
#             optimizer.zero_grad()
#             output = model(var_X_batch)
#             # regul_term = model.get_regul_term()
#             loss =loss_fn(output, var_y_batch) 
#             loss.backward()
#             optimizer.step()

#             if batch_idx%10==0:
#                 with torch.no_grad():
#                     model.projector.projection_L.data = custom_projection_L_RSRP(model.projector.projection_L,narrow_beam_CB,narrow_beam_CB_inv)
           
#             train_loss += loss.detach().item()
#             train_acc += (output.argmax(dim=1) == var_y_batch).sum().item()/var_y_batch.shape[0]
#         train_loss /= batch_idx + 1
#         train_acc /= batch_idx + 1
#         model.eval()
#         val_loss = 0
#         val_acc = 0
#         for batch_idx, (X_batch, y_batch) in enumerate(val_loader):
#             var_X_batch = X_batch.float()
#             var_y_batch = y_batch.long()  
#             output = model(var_X_batch)
#             # regul_term = model.get_regul_term()
#             loss = loss_fn(output, var_y_batch)
#             val_loss += loss.detach().item()
#             # val_acc += (output.argmax(dim=1) == var_y_batch).sum().item()/var_y_batch.shape[0]
#             _,topkbeamind=torch.topk(output,top_k,dim=1)
#             val_acc += torch.sum(torch.any(topkbeamind == var_y_batch.unsqueeze(1),dim=1)).item()/var_y_batch.shape[0]
#         val_loss /= batch_idx + 1
#         val_acc /= batch_idx + 1
#         train_loss_hist.append(train_loss)
#         val_loss_hist.append(val_loss)
#         train_acc_hist.append(train_acc)
#         val_acc_hist.append(val_acc)
#         if epoch % 10 == 0:
#             print('Epoch : {}, Top_{}, Training loss = {:.2f}, Training Acc = {:.2f}, Validation loss = {:.2f}, Validation Acc = {:.2f}.'.format(epoch,top_k,train_loss,train_acc,val_loss,val_acc))
#     return train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist


# class CNN_Classifier(nn.Module):
#     def __init__(self, n_wide_beam, n_narrow_beam):
#         super(CNN_Classifier, self).__init__()
#         self.relu = nn.ReLU()
#         self.conv1 = nn.Conv1d(in_channels=1,out_channels=16,kernel_size=3)
#         self.conv2 = nn.Conv1d(in_channels=16,out_channels=32,kernel_size=3)
#         self.conv3 = nn.Conv1d(in_channels=32,out_channels=64,kernel_size=3)
#         self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
#         output_dim = conv_dimension(input_dim=n_wide_beam*2,kernel_size=3)
#         output_dim = conv_dimension(input_dim=output_dim,kernel_size=3)
#         output_dim = conv_dimension(input_dim=output_dim,kernel_size=3)
#         output_dim = conv_dimension(input_dim=output_dim,kernel_size=2,stride=2)
#         self.flatten_dim = 64*int(output_dim)
#         self.dense1 = nn.Linear(in_features=self.flatten_dim,out_features=n_narrow_beam)
    
#     def forward(self, x):
#         x = self.relu(self.conv1(x))
#         x = self.relu(self.conv2(x))
#         x = self.pool(self.relu(self.conv3(x)))        
#         x = torch.flatten(x, 1) # flatten all dimensions except batch
#         x = self.dense1(x)
#         return x
        
# class Beam_Classifier_CNN(nn.Module):
#     def __init__(self, n_antenna, n_wide_beam, n_narrow_beam, trainable_codebook = True, theta = None, complex_codebook=None, noise_power = 0.0, norm_factor = 1.0):
#         super(Beam_Classifier_CNN, self).__init__()
#         self.trainable_codebook = trainable_codebook
#         self.n_antenna = n_antenna
#         self.n_wide_beam = n_wide_beam
#         self.n_narrow_beam = n_narrow_beam
#         self.noise_power = float(noise_power)
#         self.norm_factor = float(norm_factor)
#         if trainable_codebook:
#             self.codebook = PhaseShifter(in_features=2*n_antenna, out_features=n_wide_beam, scale=np.sqrt(n_antenna), theta=theta)
#         else:            
#             self.complex_codebook = complex_codebook # n_beams x n_antenna
#             cb_blockmatrix = codebook_blockmatrix(self.complex_codebook.T)
#             self.codebook = torch.from_numpy(cb_blockmatrix).float()
#             self.codebook.requires_grad = False
            
#         self.compute_power = ComputePower(2*n_wide_beam)
#         self.CNN_Classifier = CNN_Classifier(n_wide_beam, n_narrow_beam)
#         self.softmax = nn.Softmax()
        
#     def forward(self, x):
#         if self.trainable_codebook:
#             bf_signal = self.codebook(x)
#         else:
#             bf_signal = torch.matmul(x,self.codebook)
#         noise_vec = torch.normal(0,1, size=bf_signal.size(), device=bf_signal.device)*torch.sqrt(torch.tensor([self.noise_power/2], device=bf_signal.device))/torch.tensor([self.norm_factor], device=bf_signal.device)
#         bf_signal = bf_signal + noise_vec
#         bf_signal = bf_signal.unsqueeze(1)
#         out = self.CNN_Classifier(bf_signal)
#         return out
#     def get_codebook(self) -> np.ndarray:
#         if self.trainable_codebook:
#             return self.codebook.get_weights().detach().clone().cpu().numpy()
#         else:
#             return self.complex_codebook  

# class RSRP_CNN_Classifier(nn.Module):
#     def __init__(self, n_wide_beam, n_narrow_beam):
#         super(RSRP_CNN_Classifier, self).__init__()
#         self.relu = nn.ReLU()
#         self.conv1 = nn.Conv1d(in_channels=1,out_channels=32,kernel_size=2)
#         self.conv2 = nn.Conv1d(in_channels=32,out_channels=64,kernel_size=2)
#         self.pool = nn.MaxPool1d(kernel_size=2, stride=1)
#         output_dim = conv_dimension(input_dim=n_wide_beam,kernel_size=2)
#         output_dim = conv_dimension(input_dim=output_dim,kernel_size=2,stride=1)
#         output_dim = conv_dimension(input_dim=output_dim,kernel_size=2)
#         output_dim = conv_dimension(input_dim=output_dim,kernel_size=2,stride=1)
#         self.flatten_dim = 64*int(output_dim)
#         self.dense1 = nn.Linear(in_features=self.flatten_dim,out_features=n_narrow_beam)
    
#     def forward(self, x):
#         x = self.pool(self.relu(self.conv1(x)))        
#         x = self.pool(self.relu(self.conv2(x)))        
#         x = torch.flatten(x, 1) # flatten all dimensions except batch
#         x = self.dense1(x)
#         return x

# class Beam_Classifier_CNN_RSRP(nn.Module):
#     def __init__(self, n_antenna, n_wide_beam, n_narrow_beam, trainable_codebook = True, theta = None, complex_codebook=None, noise_power = 0.0, norm_factor = 1.0):
#         super(Beam_Classifier_CNN_RSRP, self).__init__()
#         self.trainable_codebook = trainable_codebook
#         self.n_antenna = n_antenna
#         self.n_wide_beam = n_wide_beam
#         self.n_narrow_beam = n_narrow_beam
#         self.noise_power = float(noise_power)
#         self.norm_factor = float(norm_factor)
#         if trainable_codebook:
#             self.codebook = PhaseShifter(in_features=2*n_antenna, out_features=n_wide_beam, scale=np.sqrt(n_antenna), theta=theta)
#         else:            
#             self.complex_codebook = complex_codebook # n_beams x n_antenna
#             cb_blockmatrix = codebook_blockmatrix(self.complex_codebook.T)
#             self.codebook = torch.from_numpy(cb_blockmatrix).float()
#             self.codebook.requires_grad = False
            
#         self.compute_power = ComputePower(2*n_wide_beam)
#         self.CNN_Classifier = RSRP_CNN_Classifier(n_wide_beam, n_narrow_beam)
#         self.softmax = nn.Softmax()
        
#     def forward(self, x):
#         if self.trainable_codebook:
#             bf_signal = self.codebook(x)
#         else:
#             bf_signal = torch.matmul(x,self.codebook)
#         noise_vec = torch.normal(0,1, size=bf_signal.size(), device=bf_signal.device)*torch.sqrt(torch.tensor([self.noise_power/2], device=bf_signal.device))/torch.tensor([self.norm_factor], device=bf_signal.device)
#         bf_signal = bf_signal + noise_vec
#         bf_power = self.compute_power(bf_signal)
#         bf_power = bf_power.unsqueeze(1)
#         out = self.CNN_Classifier(bf_power)
#         return out
    
#     def get_codebook(self) -> np.ndarray:
#         if self.trainable_codebook:
#             return self.codebook.get_weights().detach().clone().cpu().numpy()
#         else:
#             return self.complex_codebook 
    

    
# class projection_trainer_RSRP(nn.Module):
#     def __init__(self, n_wide_beam, n_narrow_beam, narrow_beam_CB, noise_power = 0.0, norm_factor = 1.0):
#         super(projection_trainer_RSRP, self).__init__()
#         self.n_wide_beam = n_wide_beam
#         self.n_narrow_beam = n_narrow_beam
#         self.noise_power = float(noise_power)
#         self.norm_factor = float(norm_factor)
#         self.narrow_beam_CB = torch.as_tensor(narrow_beam_CB,dtype=torch.complex64)
#         self.compute_power = ComputePower(2*n_narrow_beam)
#         self.compute_power2 = ComputePower(2*n_wide_beam)
#         self.CNN_Classifier = CNN_Classifier(n_wide_beam, n_narrow_beam)
#         self.projector = Projection_RSRP(n_narrow_beam,n_wide_beam)
#         self.relu = nn.ReLU()
#         self.dense1 = nn.Linear(in_features=n_wide_beam, out_features=2*n_wide_beam)
#         self.dense2 = nn.Linear(in_features=2*n_wide_beam, out_features=3*n_wide_beam)
#         self.dense3 = nn.Linear(in_features=3*n_wide_beam, out_features=n_narrow_beam)
#         self.softmax = nn.Softmax()
#     def forward(self, x):
#         nb_measurement_RSRP = self.compute_power(x)
#         out = self.projector(nb_measurement_RSRP)
#         out = torch.pow(torch.abs(out),1)
#         out = self.relu(out)
#         out = self.relu(self.dense1(out))
#         out = self.relu(self.dense2(out))
#         out = self.dense3(out)
#         out = out.squeeze()
#         return out
    
#     def get_projection_L(self) -> np.ndarray:
#         return self.projector.get_weights().clone().detach().numpy()
    
#     def get_initial_L(self) -> np.ndarray:
#         return self.projector.get_init_weights().clone().detach().numpy()
    
#     def get_regul_term(self):
#         temp = self.projector.get_weights().clone().detach()
#         temp = temp.T 
#         W=torch.matmul(temp[:temp.shape[0]//2,:temp.shape[1]//2]+1j*temp[temp.shape[0]//2:,temp.shape[1]//2:], self.narrow_beam_CB)
#         regul_term = torch.norm(torch.abs(W) - torch.ones(W.shape[0],W.shape[1]))
#         return regul_term
    

# class Projection_RSRP(nn.Module):
#     def __init__(self, n_nb: int, n_wb: int):
#         super(Projection_RSRP, self).__init__()
#         self.n_nb = n_nb
#         self.n_wb = n_wb
#         self.projection_L= nn.Parameter(torch.randn(n_wb,n_nb)).T
#         self.initial_L = self.projection_L.clone().detach()

#     def forward(self, inputs: Tensor) -> Tensor:
#         output = torch.matmul(inputs, self.projection_L)
#         return output
    
#     def get_weights(self) -> np.ndarray:
#         return self.projection_L.clone().detach()
#     def get_init_weights(self) -> np.ndarray:
#         return self.initial_L.clone().detach()
    
# def custom_projection_L_RSRP(weights: nn.Parameter,narrow_beam_CB,narrow_beam_CB_inv):
#     weights=weights.to(torch.complex64)
#     w=torch.matmul(weights.T,narrow_beam_CB)
#     w_normalized = w/(torch.sqrt(torch.as_tensor(narrow_beam_CB.shape[1],dtype=torch.float32))*torch.abs(w))
#     # w_normalized = w/torch.abs(w)
#     modified_weights = torch.matmul(w_normalized, narrow_beam_CB_inv)   
#     modified_weights = torch.as_tensor(modified_weights.T,dtype=torch.float32)                            
#     return modified_weights

# def fit_projector_RSRP(model,top_k, narrow_beam_CB, narrow_beam_CB_inv, train_loader, val_loader, opt, loss_fn, EPOCHS):
#     optimizer = opt
#     train_loss_hist = []
#     val_loss_hist = []
#     train_acc_hist=[]
#     val_acc_hist=[]
#     for epoch in range(EPOCHS):
#         model.train()
#         train_loss = 0
#         train_acc = 0
#         for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
#             var_X_batch = X_batch.float()
#             var_y_batch = y_batch.long()
#             optimizer.zero_grad()
#             output = model(var_X_batch)
#             # regul_term = model.get_regul_term()
#             loss =loss_fn(output, var_y_batch) 
#             loss.backward()
#             optimizer.step()

#             if batch_idx%10==0:
#                 with torch.no_grad():
#                     model.projector.projection_L.data = custom_projection_L_RSRP(model.projector.projection_L,narrow_beam_CB,narrow_beam_CB_inv)
           
#             train_loss += loss.detach().item()
#             train_acc += (output.argmax(dim=1) == var_y_batch).sum().item()/var_y_batch.shape[0]
#         train_loss /= batch_idx + 1
#         train_acc /= batch_idx + 1
#         model.eval()
#         val_loss = 0
#         val_acc = 0
#         for batch_idx, (X_batch, y_batch) in enumerate(val_loader):
#             var_X_batch = X_batch.float()
#             var_y_batch = y_batch.long()  
#             output = model(var_X_batch)
#             # regul_term = model.get_regul_term()
#             loss = loss_fn(output, var_y_batch)
#             val_loss += loss.detach().item()
#             # val_acc += (output.argmax(dim=1) == var_y_batch).sum().item()/var_y_batch.shape[0]
#             _,topkbeamind=torch.topk(output,top_k,dim=1)
#             val_acc += torch.sum(torch.any(topkbeamind == var_y_batch.unsqueeze(1),dim=1)).item()/var_y_batch.shape[0]
#         val_loss /= batch_idx + 1
#         val_acc /= batch_idx + 1
#         train_loss_hist.append(train_loss)
#         val_loss_hist.append(val_loss)
#         train_acc_hist.append(train_acc)
#         val_acc_hist.append(val_acc)
#         if epoch % 10 == 0:
#             print('Epoch : {}, Top_{}, Training loss = {:.2f}, Training Acc = {:.2f}, Validation loss = {:.2f}, Validation Acc = {:.2f}.'.format(epoch,top_k,train_loss,train_acc,val_loss,val_acc))
#     return train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist
