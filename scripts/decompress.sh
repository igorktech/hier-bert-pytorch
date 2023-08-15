#!/bin/bash

PROJECT_DIR=$(realpath "$(dirname "${BASH_SOURCE[0]}")/../")

# Decompress each part
for part in "$PROJECT_DIR/data/datasets/train/part_"*; do
  xz -d "$part"
done

# Join to single file
cat "$PROJECT_DIR/data/datasets/train/part_"* >"$PROJECT_DIR/data/datasets/train/train.xz"

# Decompress
xz -dc "$PROJECT_DIR/data/datasets/train/train.xz" >"$PROJECT_DIR/data/datasets/train/train.txt"
