import { Expr, exprShape, shapeToString } from '../expr/index.js';
import { ShapeError } from '../error.js';

/**
 * L1 norm (sum of absolute values).
 *
 * @example
 * ```ts
 * const n = norm1(x);  // ||x||_1 = sum(|x_i|)
 * ```
 */
export function norm1(arg: Expr): Expr {
  return { kind: 'norm1', arg };
}

/**
 * L2 norm (Euclidean norm).
 *
 * @example
 * ```ts
 * const n = norm2(x);  // ||x||_2 = sqrt(sum(x_i^2))
 * ```
 */
export function norm2(arg: Expr): Expr {
  return { kind: 'norm2', arg };
}

/**
 * Infinity norm (maximum absolute value).
 *
 * @example
 * ```ts
 * const n = normInf(x);  // ||x||_inf = max(|x_i|)
 * ```
 */
export function normInf(arg: Expr): Expr {
  return { kind: 'normInf', arg };
}

/**
 * Generic p-norm.
 *
 * @example
 * ```ts
 * const n = norm(x, 1);    // L1 norm
 * const n = norm(x, 2);    // L2 norm
 * const n = norm(x, Inf);  // Infinity norm
 * ```
 */
export function norm(arg: Expr, p: 1 | 2 | typeof Infinity = 2): Expr {
  switch (p) {
    case 1:
      return norm1(arg);
    case 2:
      return norm2(arg);
    case Infinity:
      return normInf(arg);
    default:
      throw new Error(`Unsupported norm p=${p}. Use 1, 2, or Infinity.`);
  }
}

/**
 * Element-wise absolute value.
 *
 * @example
 * ```ts
 * const y = abs(x);  // |x|
 * ```
 */
export function abs(arg: Expr): Expr {
  return { kind: 'abs', arg };
}

/**
 * Positive part: max(x, 0).
 *
 * @example
 * ```ts
 * const y = pos(x);  // max(x, 0)
 * ```
 */
export function pos(arg: Expr): Expr {
  return { kind: 'pos', arg };
}

/**
 * Negative part: max(-x, 0).
 *
 * This is the element-wise function neg(x) = max(-x, 0),
 * which extracts the negative portion of x.
 *
 * @example
 * ```ts
 * const y = negPart(x);  // max(-x, 0)
 * ```
 */
export function negPart(arg: Expr): Expr {
  return { kind: 'negPart', arg };
}

/**
 * Element-wise maximum of expressions.
 *
 * @example
 * ```ts
 * const z = maximum(x, y);     // Element-wise max
 * const z = maximum(x, y, z);  // Element-wise max of 3
 * ```
 */
export function maximum(...args: Expr[]): Expr {
  if (args.length === 0) {
    throw new Error('maximum requires at least one argument');
  }
  if (args.length === 1) {
    return args[0]!;
  }
  return { kind: 'maximum', args };
}

/**
 * Element-wise minimum of expressions.
 *
 * @example
 * ```ts
 * const z = minimum(x, y);     // Element-wise min
 * const z = minimum(x, y, z);  // Element-wise min of 3
 * ```
 */
export function minimum(...args: Expr[]): Expr {
  if (args.length === 0) {
    throw new Error('minimum requires at least one argument');
  }
  if (args.length === 1) {
    return args[0]!;
  }
  return { kind: 'minimum', args };
}

/**
 * Sum of squares: ||x||_2^2 = sum(x_i^2).
 *
 * @example
 * ```ts
 * const s = sumSquares(x);  // ||x||_2^2
 * ```
 */
export function sumSquares(arg: Expr): Expr {
  return { kind: 'sumSquares', arg };
}

/**
 * Quadratic form: x' P x.
 *
 * @example
 * ```ts
 * const q = quadForm(x, P);  // x' * P * x
 * ```
 */
export function quadForm(x: Expr, P: Expr): Expr {
  // Validate shapes
  const xShape = exprShape(x);
  const PShape = exprShape(P);

  if (xShape.dims.length !== 1) {
    throw new ShapeError('quadForm requires vector x', 'vector', shapeToString(xShape));
  }

  if (PShape.dims.length !== 2) {
    throw new ShapeError('quadForm requires matrix P', 'matrix', shapeToString(PShape));
  }

  const n = xShape.dims[0]!;
  if (PShape.dims[0] !== n || PShape.dims[1] !== n) {
    throw new ShapeError(
      'quadForm requires P to be n x n where x is n',
      `${n}x${n}`,
      shapeToString(PShape)
    );
  }

  // Note: For DCP compliance, P must be constant and PSD
  // This is checked during curvature analysis

  return { kind: 'quadForm', x, P };
}

/**
 * Quadratic over linear: ||x||^2 / y.
 *
 * @example
 * ```ts
 * const q = quadOverLin(x, y);  // ||x||^2 / y
 * ```
 */
export function quadOverLin(x: Expr, y: Expr): Expr {
  return { kind: 'quadOverLin', x, y };
}

/**
 * Element-wise exponential: e^x.
 *
 * This is a convex function. For DCP compliance, the argument must be affine.
 *
 * @example
 * ```ts
 * const y = exp(x);  // e^x
 * ```
 */
export function exp(arg: Expr): Expr {
  return { kind: 'exp', arg };
}

/**
 * Element-wise natural logarithm: log(x).
 *
 * This is a concave function. For DCP compliance, the argument must be affine
 * and positive.
 *
 * @example
 * ```ts
 * const y = log(x);  // ln(x)
 * ```
 */
export function log(arg: Expr): Expr {
  return { kind: 'log', arg };
}

/**
 * Element-wise entropy: -x * log(x).
 *
 * This is a concave function. For DCP compliance, the argument must be affine
 * and positive.
 *
 * @example
 * ```ts
 * const y = entropy(x);  // -x * log(x)
 * ```
 */
export function entropy(arg: Expr): Expr {
  return { kind: 'entropy', arg };
}

/**
 * Element-wise square root: sqrt(x).
 *
 * This is a concave function. For DCP compliance, the argument must be affine
 * and nonnegative.
 *
 * @example
 * ```ts
 * const y = sqrt(x);  // sqrt(x)
 * ```
 */
export function sqrt(arg: Expr): Expr {
  return { kind: 'sqrt', arg };
}

/**
 * Element-wise power: x^p.
 *
 * Curvature depends on p:
 * - p >= 1 or p < 0: Convex (for x >= 0)
 * - 0 < p < 1: Concave (for x >= 0)
 *
 * For DCP compliance, the argument must be affine and nonnegative.
 *
 * @example
 * ```ts
 * const y = power(x, 0.5);  // x^0.5 = sqrt(x), concave
 * const y = power(x, 2);    // x^2, convex
 * const y = power(x, -1);   // x^(-1) = 1/x, convex
 * ```
 */
export function power(arg: Expr, p: number): Expr {
  // Special case: p = 1 is identity
  if (p === 1) {
    return arg;
  }
  // Special case: p = 0 is constant 1 (handled during evaluation)
  return { kind: 'power', arg, p };
}
