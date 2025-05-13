#!bin/bash
# # # DROPOUT_LIST=(0 0.5)
# # # AUGMENT_LIST=(0 0.5)
# # # MODELS=("VIT")

# # # for MODEL in "${MODELS[@]}"; do
# # #   for DROPOUT in "${DROPOUT_LIST[@]}"; do
# # #     for AUGMENT in "${AUGMENT_LIST[@]}"; do

# # #         echo "==============================================="
# # #         echo "Running $MODEL | Dropout=$DROPOUT | Augment=$AUGMENT"
# # #         echo "==============================================="

# # #         DROPOUT_DECIMAL=$(printf "%.2f" "$DROPOUT")
# # #         DROPOUT_DIGITS=${DROPOUT_DECIMAL#*.}

# # #         AUGMENT_DECIMAL=$(printf "%.2f" "$AUGMENT")
# # #         AUGMENT_DIGITS=${AUGMENT_DECIMAL#*.}

# # #         MODEL_PATH="./saved_models/${MODEL}_30_${DROPOUT_DIGITS}_${AUGMENT_DIGITS}.pth"
# # #         DATASET_PATH="./assignment_1_code/fdir"

# # #         # # === TEST ===
# # #         python "test_${MODEL}.py" \
# # #         --path "$DATASET_PATH" \
# # #         --model_path "$MODEL_PATH" \
# # #         --batch_size 128 \
# # #         --num_epochs 30 \
# # #         --dropout ${DROPOUT} \
# # #         --augment ${AUGMENT} \
# # #         --results_path "./saved_models"

# # #     done
# # #   done
# # # done


MODELS=("ResNet")
AUGMENT_LIST=(0 0.1 0.2 0.5)


for MODEL in "${MODELS[@]}"; do
    for AUGMENT in "${AUGMENT_LIST[@]}"; do

      echo "==============================================="
      echo "Running $MODEL | Augment=$AUGMENT"
      echo "==============================================="

      AUGMENT_DECIMAL=$(printf "%.1f" "$AUGMENT")
      AUGMENT_DIGITS=${AUGMENT_DECIMAL#*.}

      MODEL_PATH="./saved_models2/${MODEL}_30_True_${AUGMENT_DIGITS}.pth"
      DATASET_PATH="./assignment_1_code/fdir"

      # # === TEST ===
      python "test_resnet18.py" \
      --path "$DATASET_PATH" \
      --model_path "$MODEL_PATH" \
      --batch_size 128 \
      --num_epochs 30 \
      --augment ${AUGMENT} \
      --results_path "./saved_models2"

  done
done