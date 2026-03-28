#!/bin/bash

GPU_ID=0
LOG_DIR=logs
mkdir -p "$LOG_DIR"

OPENML_IDS=(4538 41143 22 1494 54 31 1489 44160 44091 44158 44157 44126 44131 44125 44123 44090 1464 37 1510 1487 1063 41168 307 1462 23 1475 12 18 182 16 32 40499 4153 40922 1503 1068 1480 1049 44 1485 1053 4134 38 300 40984 40982 40994 6332 23517 40978)
SHOTS=(2 4 8 16)
SEEDS=(42 50 38 90 12)
CONFIGS=("configs/lightgbm.yaml" "configs/lr.yaml" "configs/xgboost.yaml")



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