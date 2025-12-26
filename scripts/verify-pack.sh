#!/bin/bash
# Verify that npm pack includes required WASM files
set -e

echo "Verifying npm package contents..."

PACK_OUTPUT=$(npm pack --dry-run 2>&1)

# Check for required WASM files
REQUIRED_FILES=(
  "wasm/pkg/bundler/clarabel_wasm_bg.wasm"
  "wasm/pkg/nodejs/clarabel_wasm_bg.wasm"
  "wasm/pkg/web/clarabel_wasm_bg.wasm"
)

MISSING=0
for file in "${REQUIRED_FILES[@]}"; do
  if ! echo "$PACK_OUTPUT" | grep -q "$file"; then
    echo "ERROR: Missing required file: $file"
    MISSING=1
  fi
done

if [ $MISSING -eq 1 ]; then
  echo ""
  echo "Package verification FAILED!"
  echo "WASM files are not being included in the npm package."
  echo "Check for .gitignore files in wasm/pkg/ directories."
  exit 1
fi

echo "All required WASM files found in package."
echo "Package verification passed!"
