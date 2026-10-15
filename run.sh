python train_dft.py --n_wide_beams 12 20 --noise_power_dBm_list -90 -87 -84 -78 -75 -72 -69 --dataset_name Rosslyn_ULA
python train_dft.py --n_wide_beams 4 6 8 10 12 14 16 18 20 --noise_power_dBm_list -81  --dataset_name Rosslyn_ULA

python train_dft.py --n_wide_beams 4 6 8 10 12 14 16 18 20 --noise_power_dBm_list -81  --dataset_name O1_28B_ULA
python train_dft.py --n_wide_beams 12 20 --noise_power_dBm_list -90 -87 -84 -78 -75 -72 -69 --dataset_name O1_28B_ULA
