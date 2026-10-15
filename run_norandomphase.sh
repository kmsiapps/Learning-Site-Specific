#!/usr/bin/env bash
set -euo pipefail

# Previous full-grid launcher, kept here for reference:
# #!/usr/bin/env bash
# set -euo pipefail
#
# # No-random-phase proposed baseline artifacts for the paper figure grids.
# # Each command sees one physical GPU as cuda:0 via CUDA_VISIBLE_DEVICES.
#
# CUDA_VISIBLE_DEVICES=0 python train_norandomphase.py --device cuda --n_wide_beams 4 6 8 10 12 14 16 18 20 --noise_power_dBm_list -81 --dataset_name Rosslyn_ULA &
# PID0=$!
#
# CUDA_VISIBLE_DEVICES=1 python train_norandomphase.py --device cuda --n_wide_beams 4 6 8 10 12 14 16 18 20 --noise_power_dBm_list -81 --dataset_name O1_28B_ULA &
# PID1=$!
#
# CUDA_VISIBLE_DEVICES=2 python train_norandomphase.py --device cuda --n_wide_beams 12 --noise_power_dBm_list -90 -87 -84 -81 -78 -75 -72 -69 --dataset_name Rosslyn_ULA &
# PID2=$!
#
# CUDA_VISIBLE_DEVICES=3 python train_norandomphase.py --device cuda --n_wide_beams 12 --noise_power_dBm_list -90 -87 -84 -81 -78 -75 -72 -69 --dataset_name O1_28B_ULA &
# PID3=$!
#
# wait "$PID0"
# wait "$PID1"
# wait "$PID2"
# wait "$PID3"

# Remaining O1_28B_ULA no-random-phase jobs as of the partial run.
# Completed already: N_W=4,6,8 @ -81 dBm and N_W=12 @ -90,-87,-84 dBm.
# Duplicate omitted: N_W=12 @ -81 dBm is run once in the N_W sweep group, not again in the noise sweep group.
# Each physical GPU is exposed to its process as cuda:0.

DATASET=O1_28B_ULA
WORKERS=${WORKERS:-8}
BATCH_SIZE=${BATCH_SIZE:-500}
EPOCHS=${EPOCHS:-200}

run_one() {
  local gpu="$1"
  local nw="$2"
  local noise="$3"
  local tx=20
  local codebook="Saved Codebooks/norandomphase_magnitude_feedback_proposed_codebook_${DATASET}_${nw}_beam_probing_codebook_64_beam_noise_${noise}.0_dBm_tx_power_${tx}_dBm.npy"
  local model="Saved Projection Models/norandomphase_magnitude_feedback_projection_trainer_${DATASET}_${nw}_beam_probing_codebook_64_beam_noise_${noise}.0_dBm_tx_power_${tx}_dBm.pt"

  if [[ -f "$codebook" && -f "$model" ]]; then
    echo "[skip] GPU $gpu: N_W=$nw noise=$noise dBm already complete"
    return 0
  fi

  echo "[run]  GPU $gpu: N_W=$nw noise=$noise dBm"
  CUDA_VISIBLE_DEVICES="$gpu" python train_norandomphase.py     --device cuda     --dataset_name "$DATASET"     --n_wide_beams "$nw"     --noise_power_dBm_list "$noise"     --epochs "$EPOCHS"     --batch_size "$BATCH_SIZE"     --num_workers "$WORKERS"
}

(
  run_one 0 10 -81
  run_one 0 18 -81
  run_one 0 12 -78
) &
PID0=$!

(
  run_one 1 12 -81
  run_one 1 20 -81
  run_one 1 12 -75
) &
PID1=$!

(
  run_one 2 14 -81
  run_one 2 12 -72
) &
PID2=$!

(
  run_one 3 16 -81
  run_one 3 12 -69
) &
PID3=$!

wait "$PID0"
wait "$PID1"
wait "$PID2"
wait "$PID3"
