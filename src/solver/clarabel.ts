/**
 * Clarabel WASM solver interface.
 *
 * Handles loading the WASM module for both browser and Node.js environments,
 * and provides a typed interface to the solver.
 */

import type { CscMatrix } from '../sparse/index.js';
import type { ConeDims } from '../canon/cone-constraint.js';
import type { SolverSettings, SolveStatus } from '../problem.js';
import { SolverError } from '../error.js';

/**
 * Raw solve result from WASM.
 */
interface WasmSolveResult {
  status: string;
  obj_val: number | null;
  x: number[] | null;
  z: number[] | null; // Dual variables
  solve_time: number;
  iterations: number;
}

/**
 * WASM module interface.
 */
interface ClarabelWasm {
  solve(
    pColPtr: Uint32Array,
    pRowIdx: Uint32Array,
    pValues: Float64Array,
    q: Float64Array,
    aColPtr: Uint32Array,
    aRowIdx: Uint32Array,
    aValues: Float64Array,
    b: Float64Array,
    n: number,
    m: number,
    coneSpecJson: string,
    settingsJson: string
  ): WasmSolveResult;
  test_wasm(): string;
  version(): string;
}

// Singleton WASM instance
let wasmInstance: ClarabelWasm | null = null;
let wasmLoading: Promise<ClarabelWasm> | null = null;

/**
 * Detect if running in Node.js environment.
 */
function isNode(): boolean {
  return (
    typeof process !== 'undefined' && process.versions != null && process.versions.node != null
  );
}

/**
 * Load WASM module.
 *
 * This automatically detects the environment (browser vs Node.js) and
 * loads the appropriate WASM build.
 */
export async function loadWasm(): Promise<ClarabelWasm> {
  if (wasmInstance) {
    return wasmInstance;
  }

  if (wasmLoading) {
    return wasmLoading;
  }

  wasmLoading = (async () => {
    try {
      if (isNode()) {
        // Node.js: use the nodejs build
        const wasm = await import('clarabel-wasm-nodejs');
        wasmInstance = wasm as unknown as ClarabelWasm;
      } else {
        // Browser: use the bundler build
        const wasm = await import('clarabel-wasm');
        wasmInstance = wasm as unknown as ClarabelWasm;
      }
      return wasmInstance;
    } catch (e) {
      wasmLoading = null;
      throw new SolverError(
        `Failed to load Clarabel WASM: ${e instanceof Error ? e.message : String(e)}`
      );
    }
  })();

  return wasmLoading;
}

/**
 * Get the loaded WASM instance (throws if not loaded).
 */
export function getWasm(): ClarabelWasm {
  if (!wasmInstance) {
    throw new SolverError('WASM not loaded. Call loadWasm() first or use solveConic().');
  }
  return wasmInstance;
}

/**
 * Test if WASM is working.
 */
export async function testWasm(): Promise<string> {
  const wasm = await loadWasm();
  return wasm.test_wasm();
}

/**
 * Get Clarabel version.
 */
export async function clarabelVersion(): Promise<string> {
  const wasm = await loadWasm();
  return wasm.version();
}

/**
 * Cone specification for the solver.
 */
interface ConeSpec {
  zero: number;
  nonneg: number;
  soc: number[];
  exp: number;
  power: number[];
}

/**
 * Settings for Clarabel solver.
 */
interface ClarabelSettings {
  verbose: boolean;
  max_iter: number;
  time_limit: number;
  tol_gap_abs: number;
  tol_gap_rel: number;
}

/**
 * Result from solving a conic problem.
 */
export interface ConicSolveResult {
  status: SolveStatus;
  objVal: number | null;
  x: Float64Array | null;
  z: Float64Array | null; // Dual variables
  solveTime: number;
  iterations: number;
}

/**
 * Convert status string to SolveStatus.
 */
function parseStatus(status: string): SolveStatus {
  if (status === 'optimal') return 'optimal';
  if (status === 'infeasible') return 'infeasible';
  if (status === 'unbounded') return 'unbounded';
  if (status === 'max_iterations') return 'max_iterations';
  if (status.startsWith('error:')) return 'numerical_error';
  return 'unknown';
}

/**
 * Solve a conic optimization problem.
 *
 * minimize    (1/2) x' P x + q' x
 * subject to  A x + s = b
 *             s in K
 *
 * @param P - Quadratic cost matrix (upper triangular CSC)
 * @param q - Linear cost vector
 * @param A - Constraint matrix (CSC)
 * @param b - Constraint vector
 * @param coneDims - Cone dimensions
 * @param settings - Solver settings
 */
export async function solveConic(
  P: CscMatrix,
  q: Float64Array,
  A: CscMatrix,
  b: Float64Array,
  coneDims: ConeDims,
  settings: SolverSettings = {}
): Promise<ConicSolveResult> {
  const wasm = await loadWasm();

  const n = P.ncols;
  const m = A.nrows;

  // Build cone spec
  const coneSpec: ConeSpec = {
    zero: coneDims.zero,
    nonneg: coneDims.nonneg,
    soc: coneDims.soc,
    exp: coneDims.exp,
    power: coneDims.power,
  };

  // Build settings - note: use large finite number for time_limit since JSON can't represent Infinity
  const clarabelSettings: ClarabelSettings = {
    verbose: settings.verbose ?? false,
    max_iter: settings.maxIter ?? 200,
    time_limit: settings.timeLimit ?? 1e10, // ~317 years
    tol_gap_abs: settings.tolGapAbs ?? 1e-8,
    tol_gap_rel: settings.tolGapRel ?? 1e-8,
  };

  // Convert CSC matrices to typed arrays with correct types
  const pColPtr = new Uint32Array(P.colPtr);
  const pRowIdx = new Uint32Array(P.rowIdx);
  const pValues = new Float64Array(P.values);

  const aColPtr = new Uint32Array(A.colPtr);
  const aRowIdx = new Uint32Array(A.rowIdx);
  const aValues = new Float64Array(A.values);

  // Call WASM solver
  const result = wasm.solve(
    pColPtr,
    pRowIdx,
    pValues,
    q,
    aColPtr,
    aRowIdx,
    aValues,
    b,
    n,
    m,
    JSON.stringify(coneSpec),
    JSON.stringify(clarabelSettings)
  );

  return {
    status: parseStatus(result.status),
    objVal: result.obj_val,
    x: result.x ? new Float64Array(result.x) : null,
    z: result.z ? new Float64Array(result.z) : null,
    solveTime: result.solve_time,
    iterations: result.iterations,
  };
}
