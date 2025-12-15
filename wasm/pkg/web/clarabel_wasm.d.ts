/* tslint:disable */
/* eslint-disable */

/**
 * Solve a conic optimization problem.
 *
 * Problem form:
 *   minimize    (1/2) x' P x + q' x
 *   subject to  A x + s = b
 *               s in K
 *
 * # Arguments
 * * `p_col_ptr`, `p_row_idx`, `p_values` - P matrix (upper triangular CSC)
 * * `q` - Linear cost vector
 * * `a_col_ptr`, `a_row_idx`, `a_values` - A matrix (CSC)
 * * `b` - Constraint vector
 * * `n` - Number of variables
 * * `m` - Number of constraints
 * * `cone_spec_json` - Cone specification as JSON string
 * * `settings_json` - Solver settings as JSON string
 */
export function solve(p_col_ptr: Uint32Array, p_row_idx: Uint32Array, p_values: Float64Array, q: Float64Array, a_col_ptr: Uint32Array, a_row_idx: Uint32Array, a_values: Float64Array, b: Float64Array, n: number, m: number, cone_spec_json: string, settings_json: string): any;

/**
 * Simple test function to verify WASM is working
 */
export function test_wasm(): string;

/**
 * Get version info
 */
export function version(): string;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly solve: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number, r: number, s: number, t: number, u: number, v: number) => any;
  readonly test_wasm: () => [number, number];
  readonly version: () => [number, number];
  readonly __wbindgen_malloc: (a: number, b: number) => number;
  readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
  readonly __wbindgen_exn_store: (a: number) => void;
  readonly __externref_table_alloc: () => number;
  readonly __wbindgen_externrefs: WebAssembly.Table;
  readonly __wbindgen_free: (a: number, b: number, c: number) => void;
  readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
*
* @returns {InitOutput}
*/
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
