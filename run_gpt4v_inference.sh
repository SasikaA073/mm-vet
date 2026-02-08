#!/bin/bash

# Define paths relative to the current directory (MMVET/MM-Vet)
BASE_DIR=$(pwd)
MODEL_NAME="gpt-4-vision-preview"
DATASET_PATH="$BASE_DIR/data/mm-vet"
RESULT_PATH="$BASE_DIR/results/$MODEL_NAME"
CONFIG_PATH="$BASE_DIR/config/openai_config.yaml"

# Load API Key from the config
if [ -f "$CONFIG_PATH" ]; then
    export OPENAI_API_KEY=$(grep "OPENAI_API_KEY" "$CONFIG_PATH" | cut -d ' ' -f 2)
fi

if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY not found in $CONFIG_PATH."
    exit 1
fi

# Ensure results directory exists
mkdir -p "$RESULT_PATH"

echo "Running GPT-4V inference on MM-Vet (Environment: mmvet)..."
echo "Dataset Path: $DATASET_PATH"
echo "Result Path: $RESULT_PATH"

# Go to inference directory for the python script execution logic
cd "$BASE_DIR/inference"

# Run the inference script using the mmvet conda environment
conda run -n mmvet python gpt4v.py \
  --mmvet_path "$DATASET_PATH" \
  --result_path "$RESULT_PATH" \
  --openai_api_key "$OPENAI_API_KEY" \
  --model_name "$MODEL_NAME" \
  --image_detail "auto"

echo "Inference complete. Results saved in $RESULT_PATH"
