import { ExprId } from '../expr/index.js';
import { CscMatrix, cscScale, cscAdd, cscIdentity } from '../sparse/index.js';
import { LinExpr, linExprZero, linExprScale, linExprAdd } from './lin-expr.js';

/**
 * Key for quadratic coefficient map.
 * Encodes a pair of variable IDs.
 */
export function quadCoeffKey(varId1: ExprId, varId2: ExprId): string {
  // Normalize order so (v1, v2) and (v2, v1) map to same key
  if (varId1 <= varId2) {
    return `${varId1}:${varId2}`;
  }
  return `${varId2}:${varId1}`;
}

/**
 * Parse a quad coeff key back to variable IDs.
 */
export function parseQuadCoeffKey(key: string): [ExprId, ExprId] {
  const parts = key.split(':');
  return [parseInt(parts[0]!, 10) as ExprId, parseInt(parts[1]!, 10) as ExprId];
}

/**
 * Quadratic expression: (1/2) x' P x + q' x + c
 *
 * Used for quadratic objectives in QP problems.
 * The quadCoeffs map stores the P matrix entries:
 * - For a single variable x, quadCoeffs has one entry (x, x) -> P
 * - The Clarabel form expects (1/2) x' P x, so we store P directly
 *
 * Note: When stuffing, we need to scale P by 2 because Clarabel uses (1/2) x' P x.
 */
export interface QuadExpr {
  /** Quadratic coefficients: key = "varId1:varId2", value = coefficient matrix */
  readonly quadCoeffs: ReadonlyMap<string, CscMatrix>;
  /** Linear term: q' x */
  readonly linear: LinExpr;
  /** Constant term: c */
  readonly constant: number;
}

/**
 * Create a QuadExpr from a linear expression (no quadratic terms).
 */
export function quadExprFromLinear(linear: LinExpr): QuadExpr {
  // Extract scalar constant from linear expression
  let constant = 0;
  if (linear.rows === 1) {
    constant = linear.constant[0] ?? 0;
  }

  // Create linear part with zero constant (constant moved to QuadExpr.constant)
  const linearPart: LinExpr = {
    coeffs: linear.coeffs,
    constant: new Float64Array(linear.rows),
    rows: linear.rows,
  };

  return {
    quadCoeffs: new Map(),
    linear: linearPart,
    constant,
  };
}

/**
 * Create a QuadExpr for a pure quadratic term: x' P x for a single variable.
 */
export function quadExprQuadratic(varId: ExprId, P: CscMatrix): QuadExpr {
  const quadCoeffs = new Map<string, CscMatrix>();
  quadCoeffs.set(quadCoeffKey(varId, varId), P);

  return {
    quadCoeffs,
    linear: linExprZero(1), // Scalar linear part (zero)
    constant: 0,
  };
}

/**
 * Create a QuadExpr for sum of squares: ||x||^2 = x' I x
 */
export function quadExprSumSquares(varId: ExprId, size: number): QuadExpr {
  // ||x||^2 = x' I x where I is identity
  // For Clarabel: (1/2) x' P x = (1/2) x' (2I) x = ||x||^2
  // So we store P = 2I (scaling handled in stuffing)
  return quadExprQuadratic(varId, cscIdentity(size));
}

/**
 * Check if a QuadExpr is purely linear (no quadratic terms).
 */
export function quadExprIsLinear(q: QuadExpr): boolean {
  return q.quadCoeffs.size === 0;
}

/**
 * Add two QuadExprs.
 */
export function quadExprAdd(a: QuadExpr, b: QuadExpr): QuadExpr {
  // Add quadratic coefficients
  const quadCoeffs = new Map<string, CscMatrix>(a.quadCoeffs);
  for (const [key, coeff] of b.quadCoeffs) {
    const existing = quadCoeffs.get(key);
    if (existing) {
      quadCoeffs.set(key, cscAdd(existing, coeff));
    } else {
      quadCoeffs.set(key, coeff);
    }
  }

  return {
    quadCoeffs,
    linear: linExprAdd(a.linear, b.linear),
    constant: a.constant + b.constant,
  };
}

/**
 * Scale a QuadExpr by a scalar.
 */
export function quadExprScale(q: QuadExpr, scalar: number): QuadExpr {
  const quadCoeffs = new Map<string, CscMatrix>();
  for (const [key, coeff] of q.quadCoeffs) {
    quadCoeffs.set(key, cscScale(coeff, scalar));
  }

  return {
    quadCoeffs,
    linear: linExprScale(q.linear, scalar),
    constant: q.constant * scalar,
  };
}

/**
 * Get all variable IDs in the QuadExpr.
 */
export function quadExprVariables(q: QuadExpr): ExprId[] {
  const vars = new Set<ExprId>();

  // From linear part
  for (const varId of q.linear.coeffs.keys()) {
    vars.add(varId);
  }

  // From quadratic part
  for (const key of q.quadCoeffs.keys()) {
    const [v1, v2] = parseQuadCoeffKey(key);
    vars.add(v1);
    vars.add(v2);
  }

  return Array.from(vars).sort();
}
