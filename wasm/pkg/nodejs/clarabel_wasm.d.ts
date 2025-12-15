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
 * * `p_data` - P matrix data (upper triangular CSC: col_ptr, row_idx, values)
 * * `q` - Linear cost vector
 * * `a_data` - A matrix data (CSC: col_ptr, row_idx, values)
 * * `b` - Constraint vector
 * * `n` - Number of variables
 * * `m` - Number of constraints
 * * `cone_spec` - Cone specification as JSON string
 * * `settings` - Solver settings as JSON string
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
