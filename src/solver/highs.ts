/**
 * HiGHS WASM solver interface.
 *
 * HiGHS is a high-performance solver for linear programming (LP),
 * mixed-integer programming (MIP), and quadratic programming (QP).
 */

import type { SolverSettings, SolveStatus } from '../problem.js';
import { SolverError } from '../error.js';

/**
 * HiGHS solution column (variable info).
 * Using loose typing to match the library's various solution types.
 */
interface HighsSolutionColumn {
  Index?: number;
  Name?: string;
  Status?: string;
  Lower?: number;
  Upper?: number;
  Type?: string;
  Primal?: number;
  Dual?: number;
}

/**
 * HiGHS solution result.
 * Uses loose typing to handle different solution statuses.
 */
interface HighsSolution {
  Status: string;
  ObjectiveValue?: number;
  Columns?: Record<string, HighsSolutionColumn>;
  Rows?: Record<string, unknown>;
}

/**
 * HiGHS WASM module interface.
 */
interface HighsWasm {
  solve(problem: string, options?: Record<string, unknown>): HighsSolution;
}

// Singleton HiGHS instance
let highsInstance: HighsWasm | null = null;
let highsLoading: Promise<HighsWasm> | null = null;

/**
 * Detect if running in Node.js environment.
 */
function isNode(): boolean {
  return (
    typeof process !== 'undefined' && process.versions != null && process.versions.node != null
  );
}

/**
 * Load HiGHS WASM module.
 *
 * Automatically detects environment (browser vs Node.js).
 */
export async function loadHiGHS(): Promise<HighsWasm> {
  if (highsInstance) {
    return highsInstance;
  }

  if (highsLoading) {
    return highsLoading;
  }

  highsLoading = (async () => {
    try {
      // Dynamic import of highs
      const highsModule = await import('highs');
      const highsLoader = highsModule.default;

      if (isNode()) {
        // Node.js: load without locateFile
        highsInstance = (await highsLoader()) as unknown as HighsWasm;
      } else {
        // Browser: use CDN for WASM file
        highsInstance = (await highsLoader({
          locateFile: (file: string) => `https://lovasoa.github.io/highs-js/${file}`,
        })) as unknown as HighsWasm;
      }

      return highsInstance;
    } catch (e) {
      highsLoading = null;
      throw new SolverError(
        `Failed to load HiGHS WASM: ${e instanceof Error ? e.message : String(e)}`
      );
    }
  })();

  return highsLoading;
}

/**
 * Get the loaded HiGHS instance (throws if not loaded).
 */
export function getHiGHS(): HighsWasm {
  if (!highsInstance) {
    throw new SolverError('HiGHS not loaded. Call loadHiGHS() first or use solveLP().');
  }
  return highsInstance;
}

/**
 * Reset the HiGHS singleton instance.
 * Useful for testing or recovering from WASM errors.
 */
export function resetHiGHS(): void {
  highsInstance = null;
  highsLoading = null;
}

/**
 * Result from solving an LP/MIP problem.
 */
export interface LPSolveResult {
  status: SolveStatus;
  objVal: number | null;
  x: Float64Array | null;
  /** Map from variable name to primal value */
  primalMap: Map<string, number> | null;
  solveTime: number;
}

/**
 * Convert HiGHS status to SolveStatus.
 */
function parseHighsStatus(status: string): SolveStatus {
  const s = status.toLowerCase();
  if (s === 'optimal') return 'optimal';
  if (s === 'infeasible') return 'infeasible';
  if (s === 'unbounded') return 'unbounded';
  if (s.includes('iteration') || s.includes('limit')) return 'max_iterations';
  if (s.includes('error')) return 'numerical_error';
  return 'unknown';
}

/**
 * Build HiGHS options from solver settings.
 */
function buildHighsOptions(settings: SolverSettings): Record<string, unknown> {
  const options: Record<string, unknown> = {};

  if (settings.verbose !== undefined) {
    options.output_flag = settings.verbose;
  }
  if (settings.maxIter !== undefined) {
    options.simplex_iteration_limit = settings.maxIter;
    options.mip_max_nodes = settings.maxIter;
  }
  if (settings.timeLimit !== undefined) {
    options.time_limit = settings.timeLimit;
  }
  if (settings.tolGapAbs !== undefined) {
    options.mip_abs_gap = settings.tolGapAbs;
  }
  if (settings.tolGapRel !== undefined) {
    options.mip_rel_gap = settings.tolGapRel;
  }

  return options;
}

/**
 * Solve a linear/mixed-integer programming problem.
 *
 * @param lpString - Problem in CPLEX LP format
 * @param varNames - Ordered list of variable names matching the LP format
 * @param settings - Solver settings
 */
export async function solveLP(
  lpString: string,
  varNames: string[],
  settings: SolverSettings = {}
): Promise<LPSolveResult> {
  const startTime = performance.now();

  const highs = await loadHiGHS();
  const options = buildHighsOptions(settings);

  let result: HighsSolution;
  try {
    result = highs.solve(lpString, options);
  } catch (e) {
    // Reset HiGHS on error to allow recovery
    resetHiGHS();
    throw new SolverError(`HiGHS solver error: ${e instanceof Error ? e.message : String(e)}`);
  }

  const endTime = performance.now();
  const solveTime = (endTime - startTime) / 1000;

  const status = parseHighsStatus(result.Status);

  if (status !== 'optimal' || !result.Columns) {
    return {
      status,
      objVal: result.ObjectiveValue ?? null,
      x: null,
      primalMap: null,
      solveTime,
    };
  }

  // Extract primal values in order
  const primalMap = new Map<string, number>();
  for (const [name, col] of Object.entries(result.Columns)) {
    primalMap.set(name, col.Primal ?? 0);
  }

  // Build ordered x vector based on varNames
  const x = new Float64Array(varNames.length);
  for (let i = 0; i < varNames.length; i++) {
    const name = varNames[i]!;
    x[i] = primalMap.get(name) ?? 0;
  }

  return {
    status,
    objVal: result.ObjectiveValue ?? null,
    x,
    primalMap,
    solveTime,
  };
}
