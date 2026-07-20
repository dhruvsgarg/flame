#!/bin/bash
# Downloads and extracts FMoW-rgb v1.1 to a target directory, without
# requiring `wilds` to be installed. Called by setup_fmow.py, but safe to
# run standalone.
#
# Usage: bash download_fmow_dataset.sh <target_dir>
set -euo pipefail

TARGET_DIR="${1:?usage: bash download_fmow_dataset.sh <target_dir>}"
URL="https://worksheets.codalab.org/rest/bundles/0xaec91eb7c9d548ebb15e1b5e60f966ab/contents/blob/"
EXPECTED_BYTES=53893324800

# Idempotency: skip entirely if this target already looks populated.
if [ -f "$TARGET_DIR/rgb_metadata.csv" ] && [ -f "$TARGET_DIR/country_code_mapping.csv" ] && [ -d "$TARGET_DIR/images" ]; then
  echo "fmow data already present at $TARGET_DIR, skipping download."
  exit 0
fi

mkdir -p "$TARGET_DIR"
ARCHIVE_PATH="$(dirname "$TARGET_DIR")/fmow_v1.1_download.tmp"

echo "downloading fMoW-rgb v1.1 (~54GB) to $ARCHIVE_PATH ..."
curl -L -o "$ARCHIVE_PATH" "$URL"

ACTUAL_BYTES=$(stat -f%z "$ARCHIVE_PATH" 2>/dev/null || stat -c%s "$ARCHIVE_PATH")
if [ "$ACTUAL_BYTES" != "$EXPECTED_BYTES" ]; then
  echo "error: downloaded $ACTUAL_BYTES bytes, expected $EXPECTED_BYTES. Not extracting." >&2
  echo "partial download left at $ARCHIVE_PATH for inspection." >&2
  exit 1
fi

echo "extracting to $TARGET_DIR ..."
tar -xzf "$ARCHIVE_PATH" -C "$TARGET_DIR" --strip-components=1

rm -f "$ARCHIVE_PATH"
echo "done. row count: $(($(wc -l < "$TARGET_DIR/rgb_metadata.csv") - 1))"