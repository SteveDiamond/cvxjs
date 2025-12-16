import { ExprData, exprShape, shapeToString } from '../expr/index.js';
import { Expr } from '../expr/expr-wrapper.js';
import { ShapeError } from '../error.js';
import { toExprData } from './affine.js';

/**
 * L1 norm (sum of absolute values).
 *
 * @example
 * ```ts
 * const n = norm1(x);  // ||x||_1 = sum(|x_i|)
 * ```
 */
export function norm1(arg: ExprData | Expr): Expr {
  return new Expr({ kind: 'norm1', arg: toExprData(arg) });
}

/**
 * L2 norm (Euclidean norm).
 *
 * @example
 * ```ts
 * const n = norm2(x);  // ||x||_2 = sqrt(sum(x_i^2))
 * ```
 */
export function norm2(arg: ExprData | Expr): Expr {
  return new Expr({ kind: 'norm2', arg: toExprData(arg) });
}

/**
 * Infinity norm (maximum absolute value).
 *
 * @example
 * ```ts
 * const n = normInf(x);  // ||x||_inf = max(|x_i|)
 * ```
 */
export function normInf(arg: ExprData | Expr): Expr {
  return new Expr({ kind: 'normInf', arg: toExprData(arg) });
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
export function norm(arg: ExprData | Expr, p: 1 | 2 | typeof Infinity = 2): Expr {
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
export function abs(arg: ExprData | Expr): Expr {
  return new Expr({ kind: 'abs', arg: toExprData(arg) });
}

/**
 * Positive part: max(x, 0).
 *
 * @example
 * ```ts
 * const y = pos(x);  // max(x, 0)
 * ```
 */
export function pos(arg: ExprData | Expr): Expr {
  return new Expr({ kind: 'pos', arg: toExprData(arg) });
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
export function negPart(arg: ExprData | Expr): Expr {
  return new Expr({ kind: 'negPart', arg: toExprData(arg) });
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
export function maximum(...args: (ExprData | Expr)[]): Expr {
  if (args.length === 0) {
    throw new Error('maximum requires at least one argument');
  }
  const exprs = args.map(toExprData);
  if (exprs.length === 1) {
    return new Expr(exprs[0]!);
  }
  return new Expr({ kind: 'maximum', args: exprs });
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
export function minimum(...args: (ExprData | Expr)[]): Expr {
  if (args.length === 0) {
    throw new Error('minimum requires at least one argument');
  }
  const exprs = args.map(toExprData);
  if (exprs.length === 1) {
    return new Expr(exprs[0]!);
  }
  return new Expr({ kind: 'minimum', args: exprs });
}

/**
 * Sum of squares: ||x||_2^2 = sum(x_i^2).
 *
 * @example
 * ```ts
 * const s = sumSquares(x);  // ||x||_2^2
 * ```
 */
export function sumSquares(arg: ExprData | Expr): Expr {
  return new Expr({ kind: 'sumSquares', arg: toExprData(arg) });
}

/**
 * Quadratic form: x' P x.
 *
 * @example
 * ```ts
 * const q = quadForm(x, P);  // x' * P * x
 * ```
 */
export function quadForm(x: ExprData | Expr, P: ExprData | Expr): Expr {
  const xExpr = toExprData(x);
  const PExpr = toExprData(P);
  // Validate shapes
  const xShape = exprShape(xExpr);
  const PShape = exprShape(PExpr);

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

  return new Expr({ kind: 'quadForm', x: xExpr, P: PExpr });
}

/**
 * Quadratic over linear: ||x||^2 / y.
 *
 * @example
 * ```ts
 * const q = quadOverLin(x, y);  // ||x||^2 / y
 * ```
 */
export function quadOverLin(x: ExprData | Expr, y: ExprData | Expr): Expr {
  return new Expr({ kind: 'quadOverLin', x: toExprData(x), y: toExprData(y) });
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
export function exp(arg: ExprData | Expr): Expr {
  return new Expr({ kind: 'exp', arg: toExprData(arg) });
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
export function log(arg: ExprData | Expr): Expr {
  return new Expr({ kind: 'log', arg: toExprData(arg) });
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
export function entropy(arg: ExprData | Expr): Expr {
  return new Expr({ kind: 'entropy', arg: toExprData(arg) });
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
export function sqrt(arg: ExprData | Expr): Expr {
  return new Expr({ kind: 'sqrt', arg: toExprData(arg) });
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
export function power(arg: ExprData | Expr, p: number): Expr {
  const a = toExprData(arg);
  // Special case: p = 1 is identity
  if (p === 1) {
    return new Expr(a);
  }
  // Special case: p = 0 is constant 1 (handled during evaluation)
  return new Expr({ kind: 'power', arg: a, p });
}
