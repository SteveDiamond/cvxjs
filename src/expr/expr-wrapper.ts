import { Expr, ExprId, exprShape, IndexRange, newExprId } from './expression.js';
import { Shape } from './shape.js';
import { toArrayData } from './constant.js';
import { Constraint } from '../constraints/constraint.js';

// Import atom functions - we'll use these internally
import {
  add as addFn,
  sub as subFn,
  neg as negFn,
  mul as mulFn,
  div as divFn,
  matmul as matmulFn,
  sum as sumFn,
  reshape as reshapeFn,
  index as indexFn,
  vstack as vstackFn,
  hstack as hstackFn,
  transpose as transposeFn,
  trace as traceFn,
  diag as diagFn,
  dot as dotFn,
  cumsum as cumsumFn,
} from '../atoms/affine.js';

import {
  norm1 as norm1Fn,
  norm2 as norm2Fn,
  normInf as normInfFn,
  abs as absFn,
  pos as posFn,
  negPart as negPartFn,
  maximum as maximumFn,
  minimum as minimumFn,
  sumSquares as sumSquaresFn,
  quadForm as quadFormFn,
  quadOverLin as quadOverLinFn,
  exp as expFn,
  log as logFn,
  entropy as entropyFn,
  sqrt as sqrtFn,
  power as powerFn,
} from '../atoms/nonlinear.js';

import {
  eq as eqFn,
  le as leFn,
  ge as geFn,
  soc as socFn,
} from '../constraints/constraint.js';

/**
 * Input type for Expression methods - accepts Expression, Expr, or number.
 */
export type ExprInput = Expression | Expr | number;

/**
 * Input type for array-like values.
 */
export type ArrayInput =
  | number
  | readonly number[]
  | readonly (readonly number[])[]
  | Float64Array;

/**
 * Create a constant Expr from a value.
 */
function makeConstant(value: ArrayInput): Expr {
  return {
    kind: 'constant',
    id: newExprId(),
    value: toArrayData(value),
  };
}

/**
 * Convert ExprInput to Expr, auto-wrapping numbers as constants.
 */
function toExpr(value: ExprInput): Expr {
  if (value instanceof Expression) {
    return value.expr;
  }
  if (typeof value === 'number') {
    return makeConstant(value);
  }
  return value;
}

/**
 * Convert array input to Expr constant.
 */
function arrayToExpr(value: ArrayInput): Expr {
  return makeConstant(value);
}

/**
 * Expression wrapper class providing fluent method chaining.
 *
 * This class wraps the underlying `Expr` discriminated union and provides
 * convenient methods for building optimization expressions.
 *
 * @example
 * ```ts
 * const x = variable(5);  // Returns Expression
 * const residual = A.matmul(x).sub(b);
 * const objective = residual.norm2().add(x.norm1().mul(lambda));
 *
 * const solution = await Problem.minimize(objective)
 *   .subjectTo([x.ge(0), x.sum().le(10)])
 *   .solve();
 * ```
 */
export class Expression {
  /**
   * The underlying expression (discriminated union).
   * Access this for interop with existing code that expects `Expr`.
   */
  public readonly expr: Expr;

  constructor(expr: Expr) {
    this.expr = expr;
  }

  // ==================== Properties ====================

  /**
   * Get the expression ID (for variables).
   * Throws if this is not a variable expression.
   */
  get id(): ExprId {
    if (this.expr.kind !== 'variable') {
      throw new Error(`Cannot get id from non-variable expression (kind: ${this.expr.kind})`);
    }
    return this.expr.id;
  }

  /**
   * Get the shape of this expression.
   */
  get shape(): Shape {
    return exprShape(this.expr);
  }

  /**
   * Get the kind of this expression.
   */
  get kind(): Expr['kind'] {
    return this.expr.kind;
  }

  // ==================== Arithmetic Operations ====================

  /**
   * Add another expression to this one.
   *
   * @example
   * ```ts
   * x.add(y)       // x + y
   * x.add(5)       // x + 5 (auto-wraps constant)
   * ```
   */
  add(other: ExprInput): Expression {
    return addFn(this.expr, toExpr(other));
  }

  /**
   * Subtract another expression from this one.
   *
   * @example
   * ```ts
   * x.sub(y)       // x - y
   * x.sub(5)       // x - 5
   * ```
   */
  sub(other: ExprInput): Expression {
    return subFn(this.expr, toExpr(other));
  }

  /**
   * Negate this expression.
   *
   * @example
   * ```ts
   * x.neg()        // -x
   * ```
   */
  neg(): Expression {
    return negFn(this.expr);
  }

  /**
   * Element-wise multiply with another expression or scalar.
   *
   * @example
   * ```ts
   * x.mul(2)       // 2 * x
   * x.mul(c)       // x * c (element-wise)
   * ```
   */
  mul(other: ExprInput): Expression {
    return mulFn(this.expr, toExpr(other));
  }

  /**
   * Divide by a scalar constant.
   *
   * @example
   * ```ts
   * x.div(2)       // x / 2
   * ```
   */
  div(other: ExprInput): Expression {
    return divFn(this.expr, toExpr(other));
  }

  /**
   * Matrix multiplication with another expression.
   *
   * @example
   * ```ts
   * A.matmul(x)    // A @ x
   * ```
   */
  matmul(other: ExprInput): Expression {
    return matmulFn(this.expr, toExpr(other));
  }

  /**
   * Dot product with another vector.
   *
   * @example
   * ```ts
   * x.dot(y)       // x' * y
   * ```
   */
  dot(other: ExprInput): Expression {
    return dotFn(this.expr, toExpr(other));
  }

  // ==================== Reduction Operations ====================

  /**
   * Sum all elements, or sum along an axis.
   *
   * @example
   * ```ts
   * x.sum()        // Sum all elements
   * A.sum(0)       // Sum along rows (column sums)
   * A.sum(1)       // Sum along columns (row sums)
   * ```
   */
  sum(axis?: number): Expression {
    return sumFn(this.expr, axis);
  }

  /**
   * Cumulative sum of elements.
   *
   * @example
   * ```ts
   * x.cumsum()     // [x1, x1+x2, x1+x2+x3, ...]
   * ```
   */
  cumsum(axis?: number): Expression {
    return cumsumFn(this.expr, axis);
  }

  /**
   * Trace of a square matrix.
   *
   * @example
   * ```ts
   * A.trace()      // tr(A)
   * ```
   */
  trace(): Expression {
    return traceFn(this.expr);
  }

  // ==================== Shape Operations ====================

  /**
   * Reshape this expression.
   *
   * @example
   * ```ts
   * x.reshape([3, 4])   // Reshape to 3x4 matrix
   * ```
   */
  reshape(shape: readonly [number] | readonly [number, number]): Expression {
    return reshapeFn(this.expr, shape);
  }

  /**
   * Index into this expression.
   *
   * @example
   * ```ts
   * x.index(0)              // x[0] (scalar)
   * x.index([0, 3])         // x[0:3] (slice)
   * A.index(0, [1, 4])      // A[0, 1:4]
   * A.index('all', 0)       // A[:, 0] (column)
   * ```
   */
  index(...indices: (number | readonly [number, number] | 'all')[]): Expression {
    return indexFn(this.expr, ...indices);
  }

  /**
   * Transpose this matrix.
   *
   * @example
   * ```ts
   * A.T()          // A'
   * A.transpose()  // A' (alias)
   * ```
   */
  T(): Expression {
    return transposeFn(this.expr);
  }

  /**
   * Transpose this matrix (alias for T()).
   */
  transpose(): Expression {
    return this.T();
  }

  /**
   * Extract diagonal or create diagonal matrix.
   *
   * @example
   * ```ts
   * A.diag()       // Extract diagonal of A as vector
   * v.diag()       // Create diagonal matrix from vector v
   * ```
   */
  diag(): Expression {
    return diagFn(this.expr);
  }

  // ==================== Norm Operations ====================

  /**
   * L1 norm (sum of absolute values).
   *
   * @example
   * ```ts
   * x.norm1()      // ||x||_1 = sum(|x_i|)
   * ```
   */
  norm1(): Expression {
    return norm1Fn(this.expr);
  }

  /**
   * L2 norm (Euclidean norm).
   *
   * @example
   * ```ts
   * x.norm2()      // ||x||_2 = sqrt(sum(x_i^2))
   * ```
   */
  norm2(): Expression {
    return norm2Fn(this.expr);
  }

  /**
   * Infinity norm (maximum absolute value).
   *
   * @example
   * ```ts
   * x.normInf()    // ||x||_inf = max(|x_i|)
   * ```
   */
  normInf(): Expression {
    return normInfFn(this.expr);
  }

  /**
   * Sum of squares.
   *
   * @example
   * ```ts
   * x.sumSquares() // ||x||_2^2 = sum(x_i^2)
   * ```
   */
  sumSquares(): Expression {
    return sumSquaresFn(this.expr);
  }

  /**
   * Quadratic form: x' P x.
   *
   * @example
   * ```ts
   * x.quadForm(P)  // x' * P * x
   * ```
   */
  quadForm(P: ExprInput): Expression {
    return new Expression(quadFormFn(this.expr, toExpr(P)));
  }

  /**
   * Quadratic over linear: ||x||^2 / y.
   *
   * @example
   * ```ts
   * x.quadOverLin(y)  // ||x||^2 / y
   * ```
   */
  quadOverLin(y: ExprInput): Expression {
    return new Expression(quadOverLinFn(this.expr, toExpr(y)));
  }

  // ==================== Element-wise Operations ====================

  /**
   * Element-wise absolute value.
   *
   * @example
   * ```ts
   * x.abs()        // |x|
   * ```
   */
  abs(): Expression {
    return new Expression(absFn(this.expr));
  }

  /**
   * Positive part: max(x, 0).
   *
   * @example
   * ```ts
   * x.pos()        // max(x, 0)
   * ```
   */
  pos(): Expression {
    return new Expression(posFn(this.expr));
  }

  /**
   * Negative part: max(-x, 0).
   *
   * @example
   * ```ts
   * x.negPart()    // max(-x, 0)
   * ```
   */
  negPart(): Expression {
    return new Expression(negPartFn(this.expr));
  }

  /**
   * Element-wise exponential: e^x.
   *
   * @example
   * ```ts
   * x.exp()        // e^x
   * ```
   */
  exp(): Expression {
    return new Expression(expFn(this.expr));
  }

  /**
   * Element-wise natural logarithm: log(x).
   *
   * @example
   * ```ts
   * x.log()        // ln(x)
   * ```
   */
  log(): Expression {
    return new Expression(logFn(this.expr));
  }

  /**
   * Element-wise entropy: -x * log(x).
   *
   * @example
   * ```ts
   * x.entropy()    // -x * log(x)
   * ```
   */
  entropy(): Expression {
    return new Expression(entropyFn(this.expr));
  }

  /**
   * Element-wise square root: sqrt(x).
   *
   * @example
   * ```ts
   * x.sqrt()       // sqrt(x)
   * ```
   */
  sqrt(): Expression {
    return new Expression(sqrtFn(this.expr));
  }

  /**
   * Element-wise power: x^p.
   *
   * @example
   * ```ts
   * x.power(2)     // x^2
   * x.power(0.5)   // sqrt(x)
   * ```
   */
  power(p: number): Expression {
    return new Expression(powerFn(this.expr, p));
  }

  // ==================== Constraint Methods ====================

  /**
   * Create equality constraint: this == other.
   *
   * @example
   * ```ts
   * x.eq(1)        // x == 1
   * sum(x).eq(10)  // sum(x) == 10
   * ```
   */
  eq(other: ExprInput): Constraint {
    return eqFn(this.expr, toExpr(other));
  }

  /**
   * Create inequality constraint: this <= other.
   *
   * @example
   * ```ts
   * x.le(10)       // x <= 10
   * norm2(x).le(1) // ||x||_2 <= 1
   * ```
   */
  le(other: ExprInput): Constraint {
    return leFn(this.expr, toExpr(other));
  }

  /**
   * Create inequality constraint: this >= other.
   *
   * @example
   * ```ts
   * x.ge(0)        // x >= 0
   * sum(x).ge(1)   // sum(x) >= 1
   * ```
   */
  ge(other: ExprInput): Constraint {
    return geFn(this.expr, toExpr(other));
  }

  /**
   * Create second-order cone constraint: ||this||_2 <= t.
   *
   * @example
   * ```ts
   * x.soc(t)       // ||x||_2 <= t
   * ```
   */
  soc(t: ExprInput): Constraint {
    return socFn(this.expr, toExpr(t));
  }

  // ==================== Static Methods for Combining Expressions ====================

  /**
   * Vertically stack expressions.
   *
   * @example
   * ```ts
   * Expression.vstack(x, y, z)
   * ```
   */
  static vstack(...args: ExprInput[]): Expression {
    return new Expression(vstackFn(...args.map(toExpr)));
  }

  /**
   * Horizontally stack expressions.
   *
   * @example
   * ```ts
   * Expression.hstack(x, y, z)
   * ```
   */
  static hstack(...args: ExprInput[]): Expression {
    return new Expression(hstackFn(...args.map(toExpr)));
  }

  /**
   * Element-wise maximum of expressions.
   *
   * @example
   * ```ts
   * Expression.maximum(x, y, z)
   * ```
   */
  static maximum(...args: ExprInput[]): Expression {
    return new Expression(maximumFn(...args.map(toExpr)));
  }

  /**
   * Element-wise minimum of expressions.
   *
   * @example
   * ```ts
   * Expression.minimum(x, y, z)
   * ```
   */
  static minimum(...args: ExprInput[]): Expression {
    return new Expression(minimumFn(...args.map(toExpr)));
  }

  /**
   * Create an Expression from a constant value.
   *
   * @example
   * ```ts
   * Expression.constant(5)
   * Expression.constant([1, 2, 3])
   * Expression.constant([[1, 2], [3, 4]])
   * ```
   */
  static constant(value: ArrayInput): Expression {
    return new Expression(arrayToExpr(value));
  }
}

/**
 * Wrap an Expr in an Expression for fluent API access.
 * This is a convenience function for interop with existing code.
 *
 * @example
 * ```ts
 * const x = variable(5);           // Already returns Expression
 * const wrapped = wrap(someExpr);  // Wrap existing Expr
 * ```
 */
export function wrap(expr: Expr): Expression {
  return new Expression(expr);
}

/**
 * Check if a value is an Expression instance.
 */
export function isExpression(value: unknown): value is Expression {
  return value instanceof Expression;
}
