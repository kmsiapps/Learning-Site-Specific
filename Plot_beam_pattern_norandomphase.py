# -*- coding: utf-8 -*-

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from beam_utils_L import calc_beam_pattern


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, nargs="+", default=["Rosslyn_ULA", "O1_28B_ULA"],
                        choices=["Rosslyn_ULA", "O1_28B_ULA", "I3_60_ULA", "O1_28_ULA"])
    parser.add_argument("--n_wide_beams", type=int, default=12)
    parser.add_argument("--n_narrow_beams", type=int, default=64)
    parser.add_argument("--noise_power_dBm", type=float, default=-81.0)
    parser.add_argument("--n_antenna", type=int, default=64)
    parser.add_argument("--out_dir", type=str, default="fig")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def tx_power_for_dataset(dataset_name):
    if dataset_name == "O1_28B_ULA":
        return 20
    return 10


def plot_one(dataset_name, n_wb, n_nb, noise_power_dBm, n_antenna, out_dir, show=False):
    tx_power_dBm = tx_power_for_dataset(dataset_name)
    noise_key = str(float(noise_power_dBm))
    codebook_path = (
        f"./Saved Codebooks/norandomphase_magnitude_feedback_proposed_codebook_"
        f"{dataset_name}_{n_wb}_beam_probing_codebook_{n_nb}_beam_noise_"
        f"{noise_key}_dBm_tx_power_{tx_power_dBm}_dBm.npy"
    )
    if not os.path.exists(codebook_path):
        raise FileNotFoundError(codebook_path)

    codebook = np.load(codebook_path)
    fig = plt.figure(figsize=(4.0, 3.6))
    ax = fig.add_subplot(111, polar=True)

    for beam in codebook:
        phi, bf_gain = calc_beam_pattern(beam, n_antenna=n_antenna)
        ax.plot(phi, bf_gain)

    ax.grid(True)
    ax.set_rmax(1)
    ax.set_rticks([0, 0.5, 1])
    ax.set_rlabel_position(-90)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_theta_zero_location("E")
    ax.set_title("No phase aug.")
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir,
        f"pattern_norandomphase_{dataset_name}_{n_wb}_{n_nb}_noise_{noise_key}_dBm_tx_power_{tx_power_dBm}_dBm.pdf",
    )
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.08)
    print(out_path)

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    args = parse_args()
    for dataset_name in args.dataset_name:
        plot_one(
            dataset_name,
            args.n_wide_beams,
            args.n_narrow_beams,
            args.noise_power_dBm,
            args.n_antenna,
            args.out_dir,
            args.show,
        )


if __name__ == "__main__":
    main()
