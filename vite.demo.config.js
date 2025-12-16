import { defineConfig } from 'vite';
import path from 'path';
import wasm from 'vite-plugin-wasm';
import topLevelAwait from 'vite-plugin-top-level-await';

export default defineConfig({
  root: './examples/demo',
  server: {
    port: 5173,
  },
  plugins: [
    wasm(),
    topLevelAwait(),
  ],
  resolve: {
    alias: {
      'cvxjs': path.resolve(__dirname, './src/index.ts'),
      'clarabel-wasm': path.resolve(__dirname, './wasm/pkg/bundler/clarabel_wasm.js'),
    },
  },
});
