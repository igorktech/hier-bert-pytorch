#!/bin/bash

PROJECT_DIR=$(realpath "$(dirname "${BASH_SOURCE[0]}")/../")
INPUT_FILE="$PROJECT_DIR/data/datasets/train/train.txt"
COMPRESSED_FILE="$PROJECT_DIR/data/datasets/train/train.xz"

xz -c "$INPUT_FILE" > "$COMPRESSED_FILE"

# Run the split command
split -b 50M $COMPRESSED_FILE "$PROJECT_DIR/data/datasets/train/part_"

# Compress each file
for part in "$PROJECT_DIR/data/datasets/train/part_"*; do
  xz "$part"
done
