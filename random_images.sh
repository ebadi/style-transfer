#!/usr/bin/env bash
# Usage: ./random_images.sh <source_dir> <N>

SRC="${1:?Usage: $0 <source_dir> <N>}"
N="${2:?Usage: $0 <source_dir> <N>}"
DEST="data/sample_images/"

mkdir -p "$DEST"

find "$SRC" -type f -iname "*.jpg" | shuf -n "$N" | while read -r file; do
    cp "$file" "$DEST/"
done

echo "Copied $N files to $DEST/"
