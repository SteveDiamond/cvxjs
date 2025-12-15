#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "Building Clarabel WASM..."

# Build for web (bundler)
echo "Building for bundler target..."
wasm-pack build --target bundler --out-dir pkg/bundler --release

# Build for Node.js
echo "Building for nodejs target..."
wasm-pack build --target nodejs --out-dir pkg/nodejs --release

# Build for web (no bundler)
echo "Building for web target..."
wasm-pack build --target web --out-dir pkg/web --release

echo "WASM build complete!"
echo "Output directories:"
echo "  - pkg/bundler (for bundlers like webpack/esbuild)"
echo "  - pkg/nodejs (for Node.js)"
echo "  - pkg/web (for browsers without bundler)"
