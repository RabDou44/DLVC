#!/bin/bash

# === CONFIGURATION ===
DATASET_PATH="./assignment_1_code/fdir"
SAVE_PATH="./saved_models"
RESULTS_PATH="./results"
BATCH_SIZE=40000
LEARNING_RATE=0.01
GPU_ID=0
EPOCHS_LIST=(1)
DROPOUT_LIST=(true)
AUGMENT_LIST=(0.5)
MODELS=("resnet18")

# === PREPARE ===
mkdir -p "$SAVE_PATH"
mkdir -p "$RESULTS_PATH"
mkdir -p "logs"

# === MAIN LOOP ===
for MODEL in "${MODELS[@]}"; do
  for EPOCHS in "${EPOCHS_LIST[@]}"; do
    for DROPOUT in "${DROPOUT_LIST[@]}"; do
      for AUGMENT in "${AUGMENT_LIST[@]}"; do

        echo "==============================================="
        echo "Running $MODEL | Epochs=$EPOCHS | Dropout=$DROPOUT | Augment=$AUGMENT"
        echo "==============================================="

        # === TRAIN ===
        python "./train_${MODEL}.py" \
          --gpu_id "$GPU_ID" \
          --path "$DATASET_PATH" \
          --save_path "$SAVE_PATH" \
          --num_epochs "$EPOCHS" \
          --batch_size "$BATCH_SIZE" \
          --learning_rate "$LEARNING_RATE" \
          --dropout "$DROPOUT" \
          --augment "$AUGMENT"

        # === TEST ===
        MODEL_SAVE_PATH="$SAVE_PATH/${MODEL}_${EPOCHS}_${DROPOUT}_${AUGMENT}.pth"
        python "./test_${MODEL}.py" \
          --gpu_id "$GPU_ID" \
          --path "$DATASET_PATH" \
          --model_path "$MODEL_SAVE_PATH" \
          --results_path "$RESULTS_PATH" \
          --batch_size "$BATCH_SIZE" \
          --num_epochs "$EPOCHS"

        # === VISUALIZE TRAIN ===
        echo "Generating TRAIN visualization for: $MODEL in $RESULTS_PATH"
        RESULT_FILE="${RESULTS_PATH}/train_log_${MODEL}_${EPOCHS}_${DROPOUT}_${AUGMENT}.csv"
        echo "Generating visualization for: $RESULT_FILE"
        python ./assignment_1_code/viz.py "$RESULT_FILE"

        # === VISUALIZE TEST ===
        echo "Generating TEST visualization for: $MODEL in $RESULTS_PATH"
        RESULT_FILE="${RESULTS_PATH}/test_log_${MODEL}_${EPOCHS}_${DROPOUT}_${AUGMENT}.csv"
        echo "Generating visualization for: $RESULT_FILE"
        python ./assignment_1_code/viz.py "$RESULT_FILE"

      done
    done
  done
done
