#!/bin/bash

FILES_TO_BACKUP=(
  "./assignment1_code"
  "/"
  "/path/to/directory1"
)

EXCLUDES=(
  "*/path/to/exclude1/*"
  "*/path/to/exclude2/*"
)

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

# Build exclude parameters for zip
EXCLUDE_PARAMS=()
for EXCLUDE in "${EXCLUDES[@]}"; do
  EXCLUDE_PARAMS+=("-x" "$EXCLUDE")
done

zip -r "$OUTPUT_FILE" "${FILES_TO_BACKUP[@]}" "${EXCLUDE_PARAMS[@]}"

if [ $? -eq 0 ]; then
  echo "Backup successful: $OUTPUT_FILE"
else
  echo "Backup failed"
  exit 1
fi
