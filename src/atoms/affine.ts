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
import { toArrayData } from '../expr/constant.js';
import { Expression } from '../expr/expr-wrapper.js';
import { ShapeError } from '../error.js';
import { newExprId } from '../expr/expression.js';

/**
 * Add two expressions.
 *
 * @example
 * ```ts
 * const z = add(x, y);
 * ```
 */
export function add(left: Expr | Expression | number, right: Expr | Expression | number): Expression {
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

  return new Expression({ kind: 'add', left: l, right: r });
}

/**
 * Subtract two expressions.
 *
 * @example
 * ```ts
 * const z = sub(x, y);  // x - y
 * ```
 */
export function sub(left: Expr | Expression | number, right: Expr | Expression | number): Expression {
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
export function neg(arg: Expr | Expression): Expression {
  const a = toExpr(arg);
  // Optimization: double negation cancels
  if (a.kind === 'neg') {
    return new Expression(a.arg);
  }
  return new Expression({ kind: 'neg', arg: a });
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
export function mul(left: Expr | Expression | number, right: Expr | Expression | number): Expression {
  const l = toExpr(left);
  const r = toExpr(right);

  return new Expression({ kind: 'mul', left: l, right: r });
}

/**
 * Divide expression by a scalar constant.
 *
 * @example
 * ```ts
 * const z = div(x, 2);  // x / 2
 * ```
 */
export function div(left: Expr | Expression | number, right: Expr | Expression | number): Expression {
  const l = toExpr(left);
  const r = toExpr(right);

  return new Expression({ kind: 'div', left: l, right: r });
}

/**
 * Matrix multiplication.
 *
 * @example
 * ```ts
 * const z = matmul(A, x);  // A @ x
 * ```
 */
export function matmul(left: Expr | Expression, right: Expr | Expression): Expression {
  const l = toExpr(left);
  const r = toExpr(right);
  const lShape = exprShape(l);
  const rShape = exprShape(r);

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

  return new Expression({ kind: 'matmul', left: l, right: r });
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
export function sum(arg: Expr | Expression, axis?: number): Expression {
  const a = toExpr(arg);
  if (axis !== undefined) {
    const shape = exprShape(a);
    if (axis < 0 || axis >= shape.dims.length) {
      throw new Error(`Invalid axis ${axis} for shape ${shapeToString(shape)}`);
    }
  }
  return new Expression({ kind: 'sum', arg: a, axis });
}

/**
 * Reshape an expression.
 *
 * @example
 * ```ts
 * const M = reshape(x, [3, 4]);  // Reshape to 3x4 matrix
 * ```
 */
export function reshape(arg: Expr | Expression, shape: readonly [number] | readonly [number, number]): Expression {
  const a = toExpr(arg);
  const argShape = exprShape(a);
  const newShape: Shape = shape.length === 1 ? vector(shape[0]) : matrix(shape[0], shape[1]);

  if (size(argShape) !== size(newShape)) {
    throw new ShapeError(
      'Cannot reshape: total size must match',
      `${size(argShape)} elements`,
      `${size(newShape)} elements`
    );
  }

  return new Expression({ kind: 'reshape', arg: a, shape: newShape });
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
export function index(arg: Expr | Expression, ...indices: (number | readonly [number, number] | 'all')[]): Expression {
  const a = toExpr(arg);
  const ranges: IndexRange[] = indices.map((idx) => {
    if (idx === 'all') {
      return { type: 'all' };
    }
    if (typeof idx === 'number') {
      return { type: 'single', index: idx };
    }
    return { type: 'range', start: idx[0], stop: idx[1] };
  });

  return new Expression({ kind: 'index', arg: a, ranges });
}

/**
 * Vertically stack expressions.
 *
 * @example
 * ```ts
 * const M = vstack(x, y, z);  // Stack vectors vertically
 * ```
 */
export function vstack(...args: (Expr | Expression)[]): Expression {
  if (args.length === 0) {
    throw new Error('vstack requires at least one argument');
  }
  const exprs = args.map(toExpr);
  if (exprs.length === 1) {
    return new Expression(exprs[0]!);
  }
  return new Expression({ kind: 'vstack', args: exprs });
}

/**
 * Horizontally stack expressions.
 *
 * @example
 * ```ts
 * const M = hstack(x, y, z);  // Stack vectors horizontally
 * ```
 */
export function hstack(...args: (Expr | Expression)[]): Expression {
  if (args.length === 0) {
    throw new Error('hstack requires at least one argument');
  }
  const exprs = args.map(toExpr);
  if (exprs.length === 1) {
    return new Expression(exprs[0]!);
  }
  return new Expression({ kind: 'hstack', args: exprs });
}

/**
 * Transpose a matrix.
 *
 * @example
 * ```ts
 * const At = transpose(A);  // A'
 * ```
 */
export function transpose(arg: Expr | Expression): Expression {
  const a = toExpr(arg);
  // Optimization: double transpose cancels
  if (a.kind === 'transpose') {
    return new Expression(a.arg);
  }
  return new Expression({ kind: 'transpose', arg: a });
}

/**
 * Trace of a square matrix.
 *
 * @example
 * ```ts
 * const t = trace(A);  // tr(A)
 * ```
 */
export function trace(arg: Expr | Expression): Expression {
  const a = toExpr(arg);
  const shape = exprShape(a);
  if (shape.dims.length !== 2 || shape.dims[0] !== shape.dims[1]) {
    throw new ShapeError('trace requires a square matrix', 'square matrix', shapeToString(shape));
  }
  return new Expression({ kind: 'trace', arg: a });
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
export function diag(arg: Expr | Expression): Expression {
  const a = toExpr(arg);
  return new Expression({ kind: 'diag', arg: a });
}

/**
 * Cumulative sum of elements.
 *
 * For a vector [x1, x2, x3], returns [x1, x1+x2, x1+x2+x3].
 * For a matrix, sums along the specified axis (0 for columns, 1 for rows).
 *
 * @example
 * ```ts
 * const cs = cumsum(x);        // Cumulative sum of vector
 * const cs = cumsum(A, 0);     // Cumulative sum along columns
 * const cs = cumsum(A, 1);     // Cumulative sum along rows
 * ```
 */
export function cumsum(arg: Expr | Expression, axis?: number): Expression {
  const a = toExpr(arg);
  return new Expression({ kind: 'cumsum', arg: a, axis });
}

/**
 * Dot product of two vectors.
 *
 * @example
 * ```ts
 * const d = dot(x, y);  // x' * y
 * ```
 */
export function dot(left: Expr | Expression, right: Expr | Expression): Expression {
  const l = toExpr(left);
  const r = toExpr(right);
  const lShape = exprShape(l);
  const rShape = exprShape(r);

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

  return sum(mul(l, r));
}

/**
 * Convert a number, Expr, or Expression to an Expr.
 */
export function toExpr(value: Expr | Expression | number): Expr {
  if (value instanceof Expression) {
    return value.expr;
  }
  if (typeof value === 'number') {
    return {
      kind: 'constant',
      id: newExprId(),
      value: toArrayData(value),
    };
  }
  return value;
}
