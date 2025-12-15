import { LinExpr } from './lin-expr.js';

/**
 * Canonical cone constraint types.
 *
 * These are the standard forms that solvers like Clarabel accept.
 */
export type ConeConstraint =
  | { readonly kind: 'zero'; readonly a: LinExpr }        // Ax + b = 0
  | { readonly kind: 'nonneg'; readonly a: LinExpr }      // Ax + b >= 0
  | { readonly kind: 'soc'; readonly t: LinExpr; readonly x: LinExpr }  // ||x|| <= t
  | { readonly kind: 'exp'; readonly x: LinExpr; readonly y: LinExpr; readonly z: LinExpr }  // Exponential cone
  | { readonly kind: 'power'; readonly x: LinExpr; readonly y: LinExpr; readonly z: LinExpr; readonly alpha: number };  // Power cone

/**
 * Cone dimensions for solver interface.
 */
export interface ConeDims {
  /** Number of equality constraints (zero cone dimension) */
  zero: number;
  /** Number of inequality constraints (nonnegative orthant dimension) */
  nonneg: number;
  /** Dimensions of second-order cones (each element is cone dimension) */
  soc: number[];
  /** Number of exponential cones (each is 3-dimensional) */
  exp: number;
  /** Alpha values for power cones (each cone is 3-dimensional) */
  power: number[];
}

/**
 * Create empty cone dimensions.
 */
export function emptyConeDims(): ConeDims {
  return {
    zero: 0,
    nonneg: 0,
    soc: [],
    exp: 0,
    power: [],
  };
}

/**
 * Get total constraint rows from cone dimensions.
 */
export function coneDimsRows(dims: ConeDims): number {
  return (
    dims.zero +
    dims.nonneg +
    dims.soc.reduce((a, b) => a + b, 0) +
    dims.exp * 3 +
    dims.power.length * 3
  );
}
