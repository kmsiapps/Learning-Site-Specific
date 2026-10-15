#!/usr/bin/env bash
set -euo pipefail

# Regenerate the paper PDFs with the additional "No phase aug." curve.
# Usage examples:
#   ./run_plots_with_norandomphase.sh
#   DATASETS="Rosslyn_ULA O1_28B_ULA" ./run_plots_with_norandomphase.sh
#   GPU=2 DATASETS="O1_28B_ULA" ./run_plots_with_norandomphase.sh

GPU=${GPU:-0}
DATASETS=${DATASETS:-Rosslyn_ULA}

for DATASET in ${DATASETS}; do
  echo "=== Plotting ${DATASET} on GPU ${GPU} ==="
  CUDA_VISIBLE_DEVICES=${GPU} MPLBACKEND=Agg DATASET_NAME=${DATASET} \
    python Plot_Results_Acc_SNR_vs_Baselines_direct_training.py

  CUDA_VISIBLE_DEVICES=${GPU} MPLBACKEND=Agg DATASET_NAME=${DATASET} \
    python plot_vs_measurement_noise.py
done
