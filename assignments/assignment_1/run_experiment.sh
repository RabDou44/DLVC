#!/bin/bash

# === CONFIGURATION ===
DATASET_PATH="./assignment_1_code/fdir"
SAVE_PATH="./saved_models"
RESULTS_PATH="./saved_models"
BATCH_SIZE=128
LEARNING_RATE=0.01
GPU_ID=0
EPOCHS_LIST=(30)
DROPOUT_LIST=(0)
AUGMENT_LIST=(0 0.5)
MODELS=("resnet18")
# SET_MODE=("Training" "Validation" "Test")

# === PREPARE ===
mkdir -p "$SAVE_PATH"
# mkdir -p "$RESULTS_PATH"
# mkdir -p "logs"

# === MAIN LOOP ===
for MODEL in "${MODELS[@]}"; do
  for EPOCHS in "${EPOCHS_LIST[@]}"; do
    for DROPOUT in "${DROPOUT_LIST[@]}"; do
      for AUGMENT in "${AUGMENT_LIST[@]}"; do

        echo "==============================================="
        echo "Running $MODEL | Epochs=$EPOCHS | Dropout=$DROPOUT | Augment=$AUGMENT"
        echo "==============================================="

        DROPOUT_DECIMAL=$(printf "%.2f" "$DROPOUT")
        DROPOUT_DIGITS=${DROPOUT_DECIMAL#*.}

        AUGMENT_DECIMAL=$(printf "%.2f" "$AUGMENT")
        AUGMENT_DIGITS=${AUGMENT_DECIMAL#*.}

        # # === TRAIN ===
        python "./train_${MODEL}.py" \
          --gpu_id "$GPU_ID" \
          --path "$DATASET_PATH" \
          --save_path "$SAVE_PATH" \
          --num_epochs "$EPOCHS" \
          --batch_size "$BATCH_SIZE" \
          --learning_rate "$LEARNING_RATE" \
          --dropout "$DROPOUT" \
          --augment "$AUGMENT"

        # # === TEST ===
        MODEL_SAVE_PATH="${SAVE_PATH}/${MODEL}_${EPOCHS}_${DROPOUT_DIGITS}_${AUGMENT_DIGITS}.pth"
        echo "Loading model from: $MODEL_SAVE_PATH"
        python "./test_${MODEL}.py" \
          --gpu_id "$GPU_ID" \
          --path "$DATASET_PATH" \
          --model_path "$MODEL_SAVE_PATH" \
          --results_path "$RESULTS_PATH" \
          --batch_size "$BATCH_SIZE" \
          --num_epochs "$EPOCHS" 

        # # # === VISUALIZE Training ===
        # echo "Generating Training visualization for: $MODEL in $RESULTS_PATH"
        # RESULT_FILE="${RESULTS_PATH}/figures/Training_log_${MODEL}_${EPOCHS}_${DROPOUT_DIGITS}_${AUGMENT_DIGITS}.png"
        # python generate_plots.py \
        #   --path "${RESULTS_PATH}/Training_log_${MODEL}_${EPOCHS}_${DROPOUT_DIGITS}_${AUGMENT_DIGITS}.csv" \
        #   --res_path "${RESULT_FILE}"

        # # # === VISUALIZE Validation ===
        # echo "Generating Validation visualization for: $MODEL in $RESULTS_PATH" 
        # RESULT_FILE="${RESULTS_PATH}/figures/Validation_log_${MODEL}_${EPOCHS}_${DROPOUT_DIGITS}_${AUGMENT_DIGITS}.png"
        # python generate_plots.py \
        #   --path "${RESULTS_PATH}/Validation_log_${MODEL}_${EPOCHS}_${DROPOUT_DIGITS}_${AUGMENT_DIGITS}.csv" \
        #   --res_path "${RESULT_FILE}"

      done
    done
  done
done
