import { ExprId } from '../expr/index.js';
import {
  CscMatrix,
  cscIdentity,
  cscZeros,
  cscScale,
  cscAdd,
  cscVstack,
  cscFromTriplets,
  cscMulMat,
} from '../sparse/index.js';

/**
 * Linear expression: sum_i(A_i * x_i) + b
 *
 * Represents an affine function of optimization variables.
 * - coeffs: Map from variable ID to coefficient matrix A_i
 * - constant: Constant term b (flattened, column-major)
 * - rows: Number of rows in the output
 */
export interface LinExpr {
  /** Coefficient matrices for each variable: varId -> A_i */
  readonly coeffs: ReadonlyMap<ExprId, CscMatrix>;
  /** Constant term b (flattened) */
  readonly constant: Float64Array;
  /** Number of output rows */
  readonly rows: number;
}

/**
 * Create a LinExpr for a variable.
 * Result is just the variable itself: I * x + 0
 */
export function linExprVariable(varId: ExprId, size: number): LinExpr {
  const coeffs = new Map<ExprId, CscMatrix>();
  coeffs.set(varId, cscIdentity(size));
  return {
    coeffs,
    constant: new Float64Array(size),
    rows: size,
  };
}

/**
 * Create a LinExpr for a constant.
 * Result is just the constant: 0 * (no vars) + c
 */
export function linExprConstant(value: Float64Array): LinExpr {
  return {
    coeffs: new Map(),
    constant: value,
    rows: value.length,
  };
}

/**
 * Create a zero LinExpr with given size.
 */
export function linExprZero(rows: number): LinExpr {
  return {
    coeffs: new Map(),
    constant: new Float64Array(rows),
    rows,
  };
}

/**
 * Broadcast a scalar LinExpr to a given size.
 * If the LinExpr is already the target size, returns it unchanged.
 * If it's a scalar (size 1), broadcasts to target size.
 * Throws if neither condition is met.
 */
function linExprBroadcast(expr: LinExpr, targetRows: number): LinExpr {
  if (expr.rows === targetRows) {
    return expr;
  }
  if (expr.rows !== 1) {
    throw new Error(`Cannot broadcast LinExpr of size ${expr.rows} to size ${targetRows}`);
  }

  // Scalar LinExpr - broadcast to target size
  // coeffs: Each coefficient matrix goes from 1xN to targetRows x N via ones matrix
  const coeffs = new Map<ExprId, CscMatrix>();
  for (const [varId, coeff] of expr.coeffs) {
    // Original coeff is 1 x varSize, we need targetRows x varSize
    // Each row should be a copy of the single row
    const varSize = coeff.ncols;
    const rows: number[] = [];
    const cols: number[] = [];
    const vals: number[] = [];

    for (let i = 0; i < targetRows; i++) {
      // Copy the first row to row i
      for (let j = 0; j < varSize; j++) {
        for (let k = coeff.colPtr[j]!; k < coeff.colPtr[j + 1]!; k++) {
          if (coeff.rowIdx[k] === 0) {
            rows.push(i);
            cols.push(j);
            vals.push(coeff.values[k]!);
          }
        }
      }
    }
    coeffs.set(varId, cscFromTriplets(targetRows, varSize, rows, cols, vals));
  }

  // Broadcast constant: replicate the scalar value
  const constant = new Float64Array(targetRows);
  const scalarVal = expr.constant[0]!;
  for (let i = 0; i < targetRows; i++) {
    constant[i] = scalarVal;
  }

  return { coeffs, constant, rows: targetRows };
}

/**
 * Add two linear expressions with automatic scalar broadcasting.
 */
export function linExprAdd(a: LinExpr, b: LinExpr): LinExpr {
  // Handle scalar broadcasting
  const targetRows = Math.max(a.rows, b.rows);
  const aBroadcast = linExprBroadcast(a, targetRows);
  const bBroadcast = linExprBroadcast(b, targetRows);

  const coeffs = new Map<ExprId, CscMatrix>();

  // Copy coefficients from a
  for (const [varId, coeff] of aBroadcast.coeffs) {
    coeffs.set(varId, coeff);
  }

  // Add coefficients from b
  for (const [varId, coeff] of bBroadcast.coeffs) {
    if (coeffs.has(varId)) {
      coeffs.set(varId, cscAdd(coeffs.get(varId)!, coeff));
    } else {
      coeffs.set(varId, coeff);
    }
  }

  // Add constants
  const constant = new Float64Array(targetRows);
  for (let i = 0; i < targetRows; i++) {
    constant[i] = aBroadcast.constant[i]! + bBroadcast.constant[i]!;
  }

  return { coeffs, constant, rows: targetRows };
}

/**
 * Subtract two linear expressions.
 */
export function linExprSub(a: LinExpr, b: LinExpr): LinExpr {
  return linExprAdd(a, linExprNeg(b));
}

/**
 * Negate a linear expression.
 */
export function linExprNeg(a: LinExpr): LinExpr {
  const coeffs = new Map<ExprId, CscMatrix>();
  for (const [varId, coeff] of a.coeffs) {
    coeffs.set(varId, cscScale(coeff, -1));
  }

  const constant = new Float64Array(a.rows);
  for (let i = 0; i < a.rows; i++) {
    constant[i] = -a.constant[i]!;
  }

  return { coeffs, constant, rows: a.rows };
}

/**
 * Scale a linear expression by a scalar.
 */
export function linExprScale(a: LinExpr, scalar: number): LinExpr {
  if (scalar === 0) {
    return linExprZero(a.rows);
  }

  const coeffs = new Map<ExprId, CscMatrix>();
  for (const [varId, coeff] of a.coeffs) {
    coeffs.set(varId, cscScale(coeff, scalar));
  }

  const constant = new Float64Array(a.rows);
  for (let i = 0; i < a.rows; i++) {
    constant[i] = scalar * a.constant[i]!;
  }

  return { coeffs, constant, rows: a.rows };
}

/**
 * Left-multiply by a constant matrix: result = M * a
 */
export function linExprLeftMul(M: CscMatrix, a: LinExpr): LinExpr {
  if (M.ncols !== a.rows) {
    throw new Error(`Matrix multiplication dimension mismatch: ${M.ncols} vs ${a.rows}`);
  }

  const coeffs = new Map<ExprId, CscMatrix>();
  for (const [varId, coeff] of a.coeffs) {
    coeffs.set(varId, cscMulMat(M, coeff));
  }

  // Multiply constant
  const constant = new Float64Array(M.nrows);
  // M * a.constant using sparse matrix-vector multiply
  for (let c = 0; c < M.ncols; c++) {
    const ac = a.constant[c]!;
    for (let i = M.colPtr[c]!; i < M.colPtr[c + 1]!; i++) {
      const row = M.rowIdx[i]!;
      constant[row] = (constant[row] ?? 0) + M.values[i]! * ac;
    }
  }

  return { coeffs, constant, rows: M.nrows };
}

/**
 * Sum all elements of a linear expression.
 */
export function linExprSum(a: LinExpr): LinExpr {
  // Create a 1 x rows sum matrix
  const sumRow: number[] = [];
  const sumCol: number[] = [];
  const sumVal: number[] = [];
  for (let i = 0; i < a.rows; i++) {
    sumRow.push(0);
    sumCol.push(i);
    sumVal.push(1);
  }
  const sumMatrix = cscFromTriplets(1, a.rows, sumRow, sumCol, sumVal);

  return linExprLeftMul(sumMatrix, a);
}

/**
 * Vertically stack linear expressions.
 */
export function linExprVstack(exprs: LinExpr[]): LinExpr {
  if (exprs.length === 0) {
    return linExprZero(0);
  }
  if (exprs.length === 1) {
    return exprs[0]!;
  }

  // Collect all variable IDs
  const allVarIds = new Set<ExprId>();
  for (const expr of exprs) {
    for (const varId of expr.coeffs.keys()) {
      allVarIds.add(varId);
    }
  }

  // Stack coefficients for each variable
  const totalRows = exprs.reduce((sum, e) => sum + e.rows, 0);
  const coeffs = new Map<ExprId, CscMatrix>();

  for (const varId of allVarIds) {
    // Get the number of columns from the first non-null coefficient
    let ncols = 0;
    for (const expr of exprs) {
      const coeff = expr.coeffs.get(varId);
      if (coeff) {
        ncols = coeff.ncols;
        break;
      }
    }

    // Stack the coefficient matrices
    let stacked: CscMatrix | null = null;
    for (const expr of exprs) {
      const coeff = expr.coeffs.get(varId) ?? cscZeros(expr.rows, ncols);
      if (stacked === null) {
        stacked = coeff;
      } else {
        stacked = cscVstack(stacked, coeff);
      }
    }

    if (stacked) {
      coeffs.set(varId, stacked);
    }
  }

  // Stack constants
  const constant = new Float64Array(totalRows);
  let offset = 0;
  for (const expr of exprs) {
    constant.set(expr.constant, offset);
    offset += expr.rows;
  }

  return { coeffs, constant, rows: totalRows };
}

/**
 * Element-wise scale a linear expression by a diagonal vector.
 * result[i] = diag[i] * a[i]
 */
export function linExprDiagScale(a: LinExpr, diag: Float64Array): LinExpr {
  if (diag.length !== a.rows) {
    throw new Error(`Diagonal scale dimension mismatch: ${diag.length} vs ${a.rows}`);
  }

  // Create diagonal matrix from diag vector
  const diagRows: number[] = [];
  const diagCols: number[] = [];
  const diagVals: number[] = [];
  for (let i = 0; i < diag.length; i++) {
    if (diag[i] !== 0) {
      diagRows.push(i);
      diagCols.push(i);
      diagVals.push(diag[i]!);
    }
  }
  const D = cscFromTriplets(a.rows, a.rows, diagRows, diagCols, diagVals);

  // D * a (left multiply by diagonal matrix)
  return linExprLeftMul(D, a);
}

/**
 * Check if linear expression is constant (no variables).
 */
export function linExprIsConstant(a: LinExpr): boolean {
  return a.coeffs.size === 0;
}

/**
 * Get the constant value of a linear expression.
 * Throws if expression has variables.
 */
export function linExprGetConstant(a: LinExpr): Float64Array {
  if (!linExprIsConstant(a)) {
    throw new Error('LinExpr is not constant');
  }
  return a.constant;
}
