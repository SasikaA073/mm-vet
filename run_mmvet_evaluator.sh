#!/bin/bash

# Define paths relative to the current directory (MMVET/MM-Vet)
BASE_DIR=$(pwd)
DATASET_PATH="$BASE_DIR/data/mm-vet"
RESULT_FILE="$BASE_DIR/results/gpt-4-vision-preview_detail-auto.json"
CONFIG_PATH="$BASE_DIR/config/openai_config.yaml"

# Load API Key from the config
if [ -f "$CONFIG_PATH" ]; then
    export OPENAI_API_KEY=$(grep "OPENAI_API_KEY" "$CONFIG_PATH" | cut -d ' ' -f 2)
fi

if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY not found in $CONFIG_PATH."
    exit 1
fi

if [ ! -f "$RESULT_FILE" ]; then
    echo "Error: Result file $RESULT_FILE not found. Please run inference first or specify the file."
    ls "$BASE_DIR/results/"
    exit 1
fi

echo "Running MM-Vet Evaluation (Environment: mmvet)..."
cd "$BASE_DIR"

# Run the evaluation script using the mmvet conda environment
conda run -n mmvet python mm-vet_evaluator.py \
    --mmvet_path "$DATASET_PATH" \
    --result_file "$RESULT_FILE" \
    --openai_api_key "$OPENAI_API_KEY" \
    --gpt_model "gpt-4-0613"

echo "Evaluation complete."
