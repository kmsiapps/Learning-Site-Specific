# -*- coding: utf-8 -*-
"""
Train the proposed magnitude-feedback baseline with the fabricated phase fixed
to zero instead of randomly sampled.
"""

import argparse
import os
import types

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from sklearn.model_selection import train_test_split

from ComplexLayers_Torch import projection_trainer_fake
from beam_utils_L import ULA_DFT_codebook as DFT_codebook


np.random.seed(8)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_wide_beams",
        type=int,
        nargs="+",
        default=[16],
        help="List of probing beams (e.g., --n_wide_beams 16 32)",
    )
    parser.add_argument(
        "--noise_power_dBm_list",
        type=float,
        nargs="+",
        default=[-81],
        help="List of noise power in dBm (e.g., --noise_power_dBm_list -81 -90)",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="O1_28B_ULA",
        choices=["O1_28B_ULA", "Rosslyn_ULA", "I3_60_ULA", "O1_28_ULA"],
        help="Dataset name",
    )
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=500, help="Training batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Training device, e.g., cuda or cpu. With CUDA_VISIBLE_DEVICES set, use cuda.",
    )
    return parser.parse_args()


def load_dataset(dataset_name):
    if dataset_name == "Rosslyn_ULA":
        h_real = np.load("./Dataset/Rosslyn/MISO_Static_FineGrid_Hmatrices_real.npy")
        h_imag = np.load("./Dataset/Rosslyn/MISO_Static_FineGrid_Hmatrices_imag.npy")
        tx_power_dBm = 10
        h = h_real + 1j * h_imag
        valid_ue_idc = np.array(
            [
                row_idx
                for (row_idx, row) in enumerate(np.concatenate((h_real, h_imag), axis=1))
                if not all(row == 0)
            ]
        )
        h = h[valid_ue_idc]
    elif dataset_name == "O1_28B_ULA":
        h = np.load("./Dataset/test_O1_28B_BS1x64x1_UE1x1x1.npy").squeeze()
        tx_power_dBm = 20
    elif dataset_name == "I3_60_ULA":
        h_real = sio.loadmat("./Dataset/DeepMIMO Dataset/I3_60_ULA/h_real.mat")["h_real"]
        h_imag = sio.loadmat("./Dataset/DeepMIMO Dataset/I3_60_ULA/h_imag.mat")["h_imag"]
        h = h_real + 1j * h_imag
        tx_power_dBm = 10
    elif dataset_name == "O1_28_ULA":
        h = np.load("./Dataset/test_O1_28_BS1x64x1_UE1x1x1.npy").squeeze()
        tx_power_dBm = 10
    else:
        raise NameError("Dataset Not Supported")

    return h, tx_power_dBm


def codebook_blockmatrix2_torch(codebook):
    real = codebook.real
    imag = codebook.imag
    return torch.cat(
        (torch.cat((real, imag), dim=1), torch.cat((-imag, real), dim=1)),
        dim=0,
    )


def custom_projection_l_device(weights, narrow_beam_cb, narrow_beam_cb_inv):
    half_rows = weights.shape[0] // 2
    half_cols = weights.shape[1] // 2
    weights_complex = weights[:half_rows, :half_cols] + 1j * weights[:half_rows, half_cols:]
    weights_complex = weights_complex.conj().T
    v = narrow_beam_cb
    w = torch.matmul(weights_complex, v)
    scale = torch.sqrt(torch.as_tensor(v.shape[1], dtype=torch.float32, device=weights.device))
    w_normalized = w / (scale * torch.abs(w).clamp_min(1e-12))

    if v.shape[1] >= v.shape[0]:
        modified_weights_complex = torch.transpose(
            torch.conj(torch.matmul(w_normalized, narrow_beam_cb_inv)), 0, 1
        )
    else:
        weights_complex_np = weights_complex.detach().cpu().numpy()
        w_normalized_np = w_normalized.detach().cpu().numpy()
        v_np = v.detach().cpu().numpy()
        modified_weights_complex_np = np.linalg.lstsq(v_np.T, w_normalized_np.T, rcond=None)[0].T
        null_proj = (
            np.identity(v_np.shape[0])
            - v_np.conj() @ np.linalg.inv(v_np.T @ v_np.conj()) @ v_np.T
        ) @ weights_complex_np.T
        modified_weights_complex_np = (modified_weights_complex_np + null_proj.T).conj().T
        modified_weights_complex = torch.as_tensor(
            modified_weights_complex_np, dtype=torch.complex64, device=weights.device
        )

    return codebook_blockmatrix2_torch(modified_weights_complex).float()


def fit_projector_device(
    model,
    top_k,
    narrow_beam_cb,
    narrow_beam_cb_inv,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    epochs,
    device,
):
    train_loss_hist = []
    val_loss_hist = []
    train_acc_hist = []
    val_acc_hist = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_acc = 0

        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.float().to(device, non_blocking=True)
            y_batch = y_batch.long().to(device, non_blocking=True)
            optimizer.zero_grad()
            output = model(x_batch)
            loss = loss_fn(output, y_batch)
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                with torch.no_grad():
                    model.projector.projection_L.data = custom_projection_l_device(
                        model.projector.projection_L,
                        narrow_beam_cb,
                        narrow_beam_cb_inv,
                    )

            train_loss += loss.detach().item()
            train_acc += (output.argmax(dim=1) == y_batch).sum().item() / y_batch.shape[0]

        train_loss /= batch_idx + 1
        train_acc /= batch_idx + 1

        model.eval()
        with torch.no_grad():
            val_loss = 0
            val_acc = 0
            for batch_idx, (x_batch, y_batch) in enumerate(val_loader):
                x_batch = x_batch.float().to(device, non_blocking=True)
                y_batch = y_batch.long().to(device, non_blocking=True)
                output = model(x_batch)
                loss = loss_fn(output, y_batch)
                val_loss += loss.detach().item()
                _, topkbeamind = torch.topk(output, top_k, dim=1)
                val_acc += torch.sum(torch.any(topkbeamind == y_batch.unsqueeze(1), dim=1)).item() / y_batch.shape[0]

            val_loss /= batch_idx + 1
            val_acc /= batch_idx + 1

        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        train_acc_hist.append(train_acc)
        val_acc_hist.append(val_acc)

        if epoch == epochs - 1:
            with torch.no_grad():
                model.projector.projection_L.data = custom_projection_l_device(
                    model.projector.projection_L,
                    narrow_beam_cb,
                    narrow_beam_cb_inv,
                )

        if epoch % 10 == 0:
            print(
                'Epoch : {}, Top_{}, Training loss = {:.2f}, Training Acc = {:.2f}, Validation loss = {:.2f}, Validation Acc = {:.2f}.'.format(
                    epoch, top_k, train_loss, train_acc, val_loss, val_acc
                )
            )

    return train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist


def fixed_zero_phase_forward(self, x):
    if self.if_fake:
        fake_x_real = x
        fake_x_imag = torch.zeros_like(x)
        fake_x = torch.cat((fake_x_real, fake_x_imag), dim=1).float()
    else:
        fake_x = x

    out = self.projector(fake_x)
    out = self.compute_power(out)
    out = self.relu(out)
    out = self.relu(self.dense1(out))
    out = self.relu(self.dense2(out))
    out = self.dense3(out)
    return out.squeeze()


def main():
    args = parse_args()

    n_wide_beams = args.n_wide_beams
    n_narrow_beams = [64 for _ in n_wide_beams]
    n_antenna = 64
    top_k = 1
    noise_factor = 0
    noiseless = False

    print("=" * 50)
    print(f"No-random-phase proposed baseline")
    print(f"Probing beams (N_W): {n_wide_beams}")
    print(f"Noise power (dBm): {args.noise_power_dBm_list}")
    print(f"Dataset name: {args.dataset_name}")
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs("./Saved Projection Models", exist_ok=True)
    os.makedirs("./Saved Projections", exist_ok=True)
    os.makedirs("./Saved Codebooks", exist_ok=True)

    h, tx_power_dBm = load_dataset(args.dataset_name)
    norm_factor = np.max(abs(h))
    h_scaled = h / norm_factor

    for noise_power_dBm in args.noise_power_dBm_list:
        if noiseless:
            noise_power_dBm = -np.inf
        noise_power = 10 ** ((noise_power_dBm - tx_power_dBm - noise_factor) / 10)

        for n_wb_i, n_wb in enumerate(n_wide_beams):
            n_nb = n_narrow_beams[n_wb_i]
            print(f"{n_wb} Wide Beams, {n_nb} Narrow Beams.")

            dft_nb_codebook = DFT_codebook(nseg=n_nb, n_antenna=n_antenna)
            dft_nb_codebook_inv = np.linalg.pinv(dft_nb_codebook)
            dft_nb_codebook_tensor = torch.as_tensor(dft_nb_codebook, dtype=torch.complex64, device=device)
            dft_nb_codebook_inv_tensor = torch.as_tensor(dft_nb_codebook_inv, dtype=torch.complex64, device=device)

            nb_measurement_complex = np.matmul(h_scaled, dft_nb_codebook.conj().T)
            noise_shape = nb_measurement_complex.shape
            noise_nb_measurement_vec = (
                np.random.normal(0, 1, noise_shape) * np.sqrt(noise_power / 2) / norm_factor
                + 1j
                * np.random.normal(0, 1, noise_shape)
                * np.sqrt(noise_power / 2)
                / norm_factor
            )
            nb_measurement_complex = nb_measurement_complex + noise_nb_measurement_vec
            nb_measurement_abs = np.abs(nb_measurement_complex)
            nb_label = np.argmax(np.power(np.absolute(nb_measurement_complex), 2), axis=1)

            train_idc, test_idc = train_test_split(np.arange(h.shape[0]), test_size=0.4)
            val_idc, _ = train_test_split(test_idc, test_size=0.5)

            x_train, y_train = nb_measurement_abs[train_idc, :], nb_label[train_idc]
            x_val, y_val = nb_measurement_abs[val_idc, :], nb_label[val_idc]

            train = torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
            val = torch.utils.data.TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
            pin_memory = device.type == "cuda"
            train_loader = torch.utils.data.DataLoader(
                train, batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory, num_workers=args.num_workers, persistent_workers=args.num_workers > 0
            )
            val_loader = torch.utils.data.DataLoader(
                val, batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory, num_workers=args.num_workers, persistent_workers=args.num_workers > 0
            )

            model = projection_trainer_fake(
                True,
                n_wide_beam=n_wb,
                n_narrow_beam=n_nb,
                narrow_beam_CB=dft_nb_codebook,
                noise_power=noise_power,
                norm_factor=norm_factor,
            )
            model.forward = types.MethodType(fixed_zero_phase_forward, model)
            model = model.to(device)

            opt = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), amsgrad=False)
            fit_projector_device(
                model,
                top_k,
                dft_nb_codebook_tensor,
                dft_nb_codebook_inv_tensor,
                train_loader,
                val_loader,
                opt,
                nn.CrossEntropyLoss(),
                args.epochs,
                device,
            )

            model_savefname = (
                "./Saved Projection Models/norandomphase_magnitude_feedback_projection_trainer_"
                f"{args.dataset_name}_{n_wb}_beam_probing_codebook_{n_nb}_beam_noise_"
                f"{noise_power_dBm}_dBm_tx_power_{tx_power_dBm}_dBm.pt"
            )
            torch.save(model.state_dict(), model_savefname)

            projection_matrix = model.projector.projection_L.detach().cpu().numpy()
            projection_savefname = (
                "./Saved Projections/norandomphase_magnitude_feedback_projection_matrix_"
                f"{args.dataset_name}_{n_wb}_beam_probing_codebook_{n_nb}_beam_noise_"
                f"{noise_power_dBm}_dBm_tx_power_{tx_power_dBm}_dBm.npy"
            )
            np.save(projection_savefname, projection_matrix.T)

            projection_matrix_block = projection_matrix.T
            projection_matrix_complex = (
                projection_matrix_block[
                    : projection_matrix_block.shape[0] // 2,
                    : projection_matrix_block.shape[1] // 2,
                ]
                + 1j
                * projection_matrix_block[
                    projection_matrix_block.shape[0] // 2 :,
                    : projection_matrix_block.shape[1] // 2,
                ]
            )
            wb_codebook = np.matmul(projection_matrix_complex.conj(), dft_nb_codebook)

            codebook_savefname = (
                "./Saved Codebooks/norandomphase_magnitude_feedback_proposed_codebook_"
                f"{args.dataset_name}_{n_wb}_beam_probing_codebook_{n_nb}_beam_noise_"
                f"{noise_power_dBm}_dBm_tx_power_{tx_power_dBm}_dBm.npy"
            )
            np.save(codebook_savefname, wb_codebook)


if __name__ == "__main__":
    main()
