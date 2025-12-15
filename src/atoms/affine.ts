import {
  Expr,
  exprShape,
  Shape,
  size,
  broadcastShape,
  shapeToString,
  IndexRange,
  matrix,
  vector,
} from '../expr/index.js';
import { constant } from '../expr/constant.js';
import { ShapeError } from '../error.js';

/**
 * Add two expressions.
 *
 * @example
 * ```ts
 * const z = add(x, y);
 * ```
 */
export function add(left: Expr | number, right: Expr | number): Expr {
  const l = toExpr(left);
  const r = toExpr(right);

  const lShape = exprShape(l);
  const rShape = exprShape(r);
  const resultShape = broadcastShape(lShape, rShape);

  if (!resultShape) {
    throw new ShapeError(
      'Cannot add expressions with incompatible shapes',
      shapeToString(lShape),
      shapeToString(rShape)
    );
  }

  return { kind: 'add', left: l, right: r };
}

/**
 * Subtract two expressions.
 *
 * @example
 * ```ts
 * const z = sub(x, y);  // x - y
 * ```
 */
export function sub(left: Expr | number, right: Expr | number): Expr {
  return add(left, neg(toExpr(right)));
}

/**
 * Negate an expression.
 *
 * @example
 * ```ts
 * const y = neg(x);  // -x
 * ```
 */
export function neg(arg: Expr): Expr {
  // Optimization: double negation cancels
  if (arg.kind === 'neg') {
    return arg.arg;
  }
  return { kind: 'neg', arg };
}

/**
 * Element-wise multiply two expressions.
 * At least one operand must be a scalar or constant.
 *
 * @example
 * ```ts
 * const z = mul(2, x);      // 2 * x
 * const z = mul(x, c);      // x * c (c is constant)
 * ```
 */
export function mul(left: Expr | number, right: Expr | number): Expr {
  const l = toExpr(left);
  const r = toExpr(right);

  return { kind: 'mul', left: l, right: r };
}

/**
 * Divide expression by a scalar constant.
 *
 * @example
 * ```ts
 * const z = div(x, 2);  // x / 2
 * ```
 */
export function div(left: Expr | number, right: Expr | number): Expr {
  const l = toExpr(left);
  const r = toExpr(right);

  return { kind: 'div', left: l, right: r };
}

/**
 * Matrix multiplication.
 *
 * @example
 * ```ts
 * const z = matmul(A, x);  // A @ x
 * ```
 */
export function matmul(left: Expr, right: Expr): Expr {
  const lShape = exprShape(left);
  const rShape = exprShape(right);

  // Validate dimensions
  const lCols = lShape.dims.length === 1 ? lShape.dims[0] : lShape.dims[1];
  const rRows = rShape.dims[0];

  if (lCols !== rRows) {
    throw new ShapeError(
      'Incompatible dimensions for matrix multiplication',
      `inner dimension ${lCols}`,
      `${rRows}`
    );
  }

  return { kind: 'matmul', left, right };
}

/**
 * Sum all elements of an expression, or sum along an axis.
 *
 * @example
 * ```ts
 * const s = sum(x);           // Sum all elements
 * const s = sum(A, 0);        // Sum along rows (column sums)
 * const s = sum(A, 1);        // Sum along columns (row sums)
 * ```
 */
export function sum(arg: Expr, axis?: number): Expr {
  if (axis !== undefined) {
    const shape = exprShape(arg);
    if (axis < 0 || axis >= shape.dims.length) {
      throw new Error(`Invalid axis ${axis} for shape ${shapeToString(shape)}`);
    }
  }
  return { kind: 'sum', arg, axis };
}

/**
 * Reshape an expression.
 *
 * @example
 * ```ts
 * const M = reshape(x, [3, 4]);  // Reshape to 3x4 matrix
 * ```
 */
export function reshape(arg: Expr, shape: readonly [number] | readonly [number, number]): Expr {
  const argShape = exprShape(arg);
  const newShape: Shape = shape.length === 1 ? vector(shape[0]) : matrix(shape[0], shape[1]);

  if (size(argShape) !== size(newShape)) {
    throw new ShapeError(
      'Cannot reshape: total size must match',
      `${size(argShape)} elements`,
      `${size(newShape)} elements`
    );
  }

  return { kind: 'reshape', arg, shape: newShape };
}

/**
 * Index into an expression.
 *
 * @example
 * ```ts
 * const y = index(x, 0);           // x[0] (scalar)
 * const y = index(x, [0, 3]);      // x[0:3] (slice)
 * const y = index(A, 0, [1, 4]);   // A[0, 1:4]
 * ```
 */
export function index(arg: Expr, ...indices: (number | readonly [number, number] | 'all')[]): Expr {
  const ranges: IndexRange[] = indices.map((idx) => {
    if (idx === 'all') {
      return { type: 'all' };
    }
    if (typeof idx === 'number') {
      return { type: 'single', index: idx };
    }
    return { type: 'range', start: idx[0], stop: idx[1] };
  });

  return { kind: 'index', arg, ranges };
}

/**
 * Vertically stack expressions.
 *
 * @example
 * ```ts
 * const M = vstack(x, y, z);  // Stack vectors vertically
 * ```
 */
export function vstack(...args: Expr[]): Expr {
  if (args.length === 0) {
    throw new Error('vstack requires at least one argument');
  }
  if (args.length === 1) {
    return args[0]!;
  }
  return { kind: 'vstack', args };
}

/**
 * Horizontally stack expressions.
 *
 * @example
 * ```ts
 * const M = hstack(x, y, z);  // Stack vectors horizontally
 * ```
 */
export function hstack(...args: Expr[]): Expr {
  if (args.length === 0) {
    throw new Error('hstack requires at least one argument');
  }
  if (args.length === 1) {
    return args[0]!;
  }
  return { kind: 'hstack', args };
}

/**
 * Transpose a matrix.
 *
 * @example
 * ```ts
 * const At = transpose(A);  // A'
 * ```
 */
export function transpose(arg: Expr): Expr {
  // Optimization: double transpose cancels
  if (arg.kind === 'transpose') {
    return arg.arg;
  }
  return { kind: 'transpose', arg };
}

/**
 * Trace of a square matrix.
 *
 * @example
 * ```ts
 * const t = trace(A);  // tr(A)
 * ```
 */
export function trace(arg: Expr): Expr {
  const shape = exprShape(arg);
  if (shape.dims.length !== 2 || shape.dims[0] !== shape.dims[1]) {
    throw new ShapeError('trace requires a square matrix', 'square matrix', shapeToString(shape));
  }
  return { kind: 'trace', arg };
}

/**
 * Extract diagonal or create diagonal matrix.
 *
 * @example
 * ```ts
 * const d = diag(A);   // Extract diagonal of A as vector
 * const D = diag(v);   // Create diagonal matrix from vector v
 * ```
 */
export function diag(arg: Expr): Expr {
  return { kind: 'diag', arg };
}

/**
 * Dot product of two vectors.
 *
 * @example
 * ```ts
 * const d = dot(x, y);  // x' * y
 * ```
 */
export function dot(left: Expr, right: Expr): Expr {
  const lShape = exprShape(left);
  const rShape = exprShape(right);

  if (lShape.dims.length !== 1 || rShape.dims.length !== 1) {
    throw new ShapeError(
      'dot requires vectors',
      'vectors',
      `${shapeToString(lShape)}, ${shapeToString(rShape)}`
    );
  }

  if (lShape.dims[0] !== rShape.dims[0]) {
    throw new ShapeError(
      'dot requires vectors of same length',
      `${lShape.dims[0]}`,
      `${rShape.dims[0]}`
    );
  }

  return sum(mul(left, right));
}

/**
 * Convert a number or Expr to an Expr.
 */
export function toExpr(value: Expr | number): Expr {
  if (typeof value === 'number') {
    return constant(value);
  }
  return value;
}
