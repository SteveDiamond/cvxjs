import { Expr } from '../expr/index.js';

/**
 * Curvature of an expression in disciplined convex programming.
 *
 * The curvature hierarchy (from most to least restrictive):
 * - Constant: Fixed value, no variables
 * - Affine: Linear in variables (both convex and concave)
 * - Convex: f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y)
 * - Concave: f(λx + (1-λ)y) ≥ λf(x) + (1-λ)f(y)
 * - Unknown: Does not satisfy DCP rules
 */
export enum Curvature {
  Constant = 'constant',
  Affine = 'affine',
  Convex = 'convex',
  Concave = 'concave',
  Unknown = 'unknown',
}

/** Check if curvature is convex (includes constant and affine) */
export function isConvex(c: Curvature): boolean {
  return c === Curvature.Constant || c === Curvature.Affine || c === Curvature.Convex;
}

/** Check if curvature is concave (includes constant and affine) */
export function isConcave(c: Curvature): boolean {
  return c === Curvature.Constant || c === Curvature.Affine || c === Curvature.Concave;
}

/** Check if curvature is affine (includes constant) */
export function isAffine(c: Curvature): boolean {
  return c === Curvature.Constant || c === Curvature.Affine;
}

/** Check if curvature is constant */
export function isConstantCurvature(c: Curvature): boolean {
  return c === Curvature.Constant;
}

/**
 * Combine curvatures under addition.
 *
 * Rules:
 * - Constant + X = X
 * - Affine + X = X
 * - Convex + Convex = Convex
 * - Concave + Concave = Concave
 * - Convex + Concave = Unknown
 */
export function addCurvature(a: Curvature, b: Curvature): Curvature {
  if (a === Curvature.Constant) return b;
  if (b === Curvature.Constant) return a;
  if (a === Curvature.Affine) return b;
  if (b === Curvature.Affine) return a;
  if (a === b) return a; // Convex + Convex = Convex, etc.
  if (a === Curvature.Unknown || b === Curvature.Unknown) return Curvature.Unknown;
  return Curvature.Unknown; // Convex + Concave = Unknown
}

/**
 * Negate a curvature.
 * Convex becomes Concave and vice versa.
 */
export function negateCurvature(c: Curvature): Curvature {
  if (c === Curvature.Convex) return Curvature.Concave;
  if (c === Curvature.Concave) return Curvature.Convex;
  return c; // Constant, Affine, Unknown unchanged
}

/**
 * Multiply curvature by a constant scalar.
 *
 * Rules:
 * - Positive scalar preserves curvature
 * - Zero scalar makes it constant
 * - Negative scalar negates curvature
 */
export function scaleCurvature(c: Curvature, scalar: number): Curvature {
  if (scalar === 0) return Curvature.Constant;
  if (scalar > 0) return c;
  return negateCurvature(c);
}

/**
 * Compute the curvature of an expression.
 *
 * This is the core DCP analysis function that determines whether
 * an expression is convex, concave, affine, constant, or unknown.
 */
export function curvature(expr: Expr): Curvature {
  switch (expr.kind) {
    // === Leaf nodes ===
    case 'variable':
      return Curvature.Affine;

    case 'constant':
      return Curvature.Constant;

    // === Affine atoms ===
    case 'add':
      return addCurvature(curvature(expr.left), curvature(expr.right));

    case 'neg':
      return negateCurvature(curvature(expr.arg));

    case 'mul': {
      // Element-wise multiplication: one operand must be constant
      const leftCurv = curvature(expr.left);
      const rightCurv = curvature(expr.right);

      if (leftCurv === Curvature.Constant) {
        // Constant * X: need to check sign of constant for proper scaling
        // For now, treat as preserving curvature (assume positive)
        // Full implementation would check sign
        return rightCurv;
      }
      if (rightCurv === Curvature.Constant) {
        return leftCurv;
      }
      // Non-constant * Non-constant is not DCP
      return Curvature.Unknown;
    }

    case 'div': {
      // Division by constant
      const leftCurv = curvature(expr.left);
      const rightCurv = curvature(expr.right);

      if (rightCurv !== Curvature.Constant) {
        return Curvature.Unknown; // Division by non-constant
      }
      return leftCurv; // Assumes positive divisor
    }

    case 'matmul': {
      // Matrix multiplication: at least one operand must be constant
      const leftCurv = curvature(expr.left);
      const rightCurv = curvature(expr.right);

      if (leftCurv === Curvature.Constant) {
        return rightCurv; // Constant @ X preserves curvature
      }
      if (rightCurv === Curvature.Constant) {
        return leftCurv; // X @ Constant preserves curvature
      }
      return Curvature.Unknown; // Variable @ Variable is not DCP
    }

    case 'sum':
    case 'reshape':
    case 'index':
    case 'transpose':
    case 'trace':
    case 'diag':
      // These affine operations preserve curvature
      return curvature(expr.arg);

    case 'vstack':
    case 'hstack': {
      // Stacking preserves curvature if all args have same curvature
      let result = Curvature.Constant;
      for (const arg of expr.args) {
        result = addCurvature(result, curvature(arg));
      }
      return result;
    }

    // === Nonlinear convex atoms ===
    case 'norm1':
    case 'norm2':
    case 'normInf':
    case 'abs': {
      // Convex atoms require affine argument for DCP
      const argCurv = curvature(expr.arg);
      if (isAffine(argCurv)) {
        return Curvature.Convex;
      }
      return Curvature.Unknown;
    }

    case 'pos': {
      // pos(x) = max(x, 0) is convex
      // DCP: argument must be affine or convex
      const argCurv = curvature(expr.arg);
      if (isAffine(argCurv)) {
        return Curvature.Convex;
      }
      if (argCurv === Curvature.Convex) {
        return Curvature.Convex; // Convex of convex is convex
      }
      return Curvature.Unknown;
    }

    case 'maximum': {
      // maximum is convex: max of convex functions is convex
      // DCP: all arguments must be convex (or affine)
      for (const arg of expr.args) {
        if (!isConvex(curvature(arg))) {
          return Curvature.Unknown;
        }
      }
      // Result is convex unless all are constant
      const allConstant = expr.args.every((arg) => curvature(arg) === Curvature.Constant);
      return allConstant ? Curvature.Constant : Curvature.Convex;
    }

    case 'sumSquares': {
      // sum_squares(x) = ||x||_2^2 is convex
      // DCP: argument must be affine
      const argCurv = curvature(expr.arg);
      if (isAffine(argCurv)) {
        return Curvature.Convex;
      }
      return Curvature.Unknown;
    }

    case 'quadForm': {
      // x'Px is convex if P is positive semidefinite
      // For DCP, x must be affine and P must be constant PSD
      // (We assume P is PSD if it's constant)
      const xCurv = curvature(expr.x);
      const PCurv = curvature(expr.P);

      if (PCurv !== Curvature.Constant) {
        return Curvature.Unknown; // P must be constant
      }
      if (isAffine(xCurv)) {
        return Curvature.Convex; // Assuming P is PSD
      }
      return Curvature.Unknown;
    }

    case 'quadOverLin': {
      // ||x||^2 / y is convex for y > 0
      // DCP: x must be affine, y must be affine and positive
      const xCurv = curvature(expr.x);
      const yCurv = curvature(expr.y);

      if (isAffine(xCurv) && isAffine(yCurv)) {
        return Curvature.Convex;
      }
      return Curvature.Unknown;
    }

    // === Nonlinear concave atoms ===
    case 'minimum': {
      // minimum is concave: min of concave functions is concave
      // DCP: all arguments must be concave (or affine)
      for (const arg of expr.args) {
        if (!isConcave(curvature(arg))) {
          return Curvature.Unknown;
        }
      }
      const allConstant = expr.args.every((arg) => curvature(arg) === Curvature.Constant);
      return allConstant ? Curvature.Constant : Curvature.Concave;
    }

    default: {
      // Exhaustiveness check
      const _exhaustive: never = expr;
      return Curvature.Unknown;
    }
  }
}

/**
 * Check if an expression is DCP-convex (can be minimized).
 */
export function isDcpConvex(expr: Expr): boolean {
  return isConvex(curvature(expr));
}

/**
 * Check if an expression is DCP-concave (can be maximized).
 */
export function isDcpConcave(expr: Expr): boolean {
  return isConcave(curvature(expr));
}

/**
 * Check if an expression is DCP-affine (can be used in equality constraints).
 */
export function isDcpAffine(expr: Expr): boolean {
  return isAffine(curvature(expr));
}
