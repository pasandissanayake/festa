#!/bin/bash

GPU_ID=0
LOG_DIR=logs
mkdir -p "$LOG_DIR"

OPENML_IDS=(4538)
SHOTS=(4)
SEEDS=(42)
CONFIGS=("configs/tabdistill.yaml")



for openml_id in "${OPENML_IDS[@]}"; do
  for shot in "${SHOTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      for config in "${CONFIGS[@]}"; do

        CONFIG_NAME=$(basename "$config" .yaml)
        EXP_NAME="id${openml_id}_shot${shot}_seed${seed}_${CONFIG_NAME}"
        LOG_FILE="$LOG_DIR/${EXP_NAME}.log"

        echo "==============================================="
        echo "Running $EXP_NAME"
        echo "Logging to $LOG_FILE"
        echo "==============================================="

        python3 main.py \
          --gpu_id $GPU_ID \
          --openml_id $openml_id \
          --shot $shot \
          --seed $seed \
          --config_filename $config \
          --force_train \
          2>&1 | tee "$LOG_FILE"

        if [ ${PIPESTATUS[0]} -ne 0 ]; then
          echo "❌ FAILED: $EXP_NAME"
        else
          echo "✅ DONE: $EXP_NAME"
        fi

      done
    done
  done
done