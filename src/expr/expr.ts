import { ExprData, ExprId, exprShape, newExprId } from './expr-data.js';
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

import {
  evaluate as evaluateFn,
  evaluateScalar as evaluateScalarFn,
  VariableValues,
} from './evaluate.js';

/**
 * Input type for Expr methods - accepts Expr, ExprData, numbers, or arrays.
 */
export type ExprInput = Expr | ExprData | number | readonly number[] | readonly (readonly number[])[];

/**
 * Input type for array-like values.
 */
export type ArrayInput =
  | number
  | readonly number[]
  | readonly (readonly number[])[]
  | Float64Array;

/**
 * Create a constant ExprData from a value.
 */
function makeConstant(value: ArrayInput): ExprData {
  return {
    kind: 'constant',
    id: newExprId(),
    value: toArrayData(value),
  };
}

/**
 * Convert ExprInput to ExprData, auto-wrapping numbers and arrays as constants.
 */
function toExprData(value: ExprInput): ExprData {
  if (value instanceof Expr) {
    return value.data;
  }
  if (typeof value === 'number') {
    return makeConstant(value);
  }
  // Check if it's an array (1D or 2D)
  if (Array.isArray(value)) {
    return makeConstant(value as ArrayInput);
  }
  // Must be ExprData
  return value as ExprData;
}

/**
 * Convert array input to ExprData constant.
 */
function arrayToExprData(value: ArrayInput): ExprData {
  return makeConstant(value);
}

/**
 * Expression wrapper class providing fluent method chaining.
 *
 * This class wraps the underlying `ExprData` discriminated union and provides
 * convenient methods for building optimization expressions.
 *
 * @example
 * ```ts
 * const x = variable(5);  // Returns Expr
 * const residual = A.matmul(x).sub(b);
 * const objective = residual.norm2().add(x.norm1().mul(lambda));
 *
 * const solution = await Problem.minimize(objective)
 *   .subjectTo([x.ge(0), x.sum().le(10)])
 *   .solve();
 * ```
 */
export class Expr {
  /**
   * The underlying expression data (discriminated union).
   * Access this for interop with existing code that expects `ExprData`.
   */
  public readonly data: ExprData;

  constructor(data: ExprData) {
    this.data = data;
  }

  // ==================== Properties ====================

  /**
   * Get the expression ID (for variables).
   * Throws if this is not a variable expression.
   */
  get id(): ExprId {
    if (this.data.kind !== 'variable') {
      throw new Error(`Cannot get id from non-variable expression (kind: ${this.data.kind})`);
    }
    return this.data.id;
  }

  /**
   * Get the shape of this expression.
   */
  get shape(): Shape {
    return exprShape(this.data);
  }

  /**
   * Get the kind of this expression.
   */
  get kind(): ExprData['kind'] {
    return this.data.kind;
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
  add(other: ExprInput): Expr {
    return addFn(this.data, toExprData(other));
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
  sub(other: ExprInput): Expr {
    return subFn(this.data, toExprData(other));
  }

  /**
   * Negate this expression.
   *
   * @example
   * ```ts
   * x.neg()        // -x
   * ```
   */
  neg(): Expr {
    return negFn(this.data);
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
  mul(other: ExprInput): Expr {
    return mulFn(this.data, toExprData(other));
  }

  /**
   * Divide by a scalar constant.
   *
   * @example
   * ```ts
   * x.div(2)       // x / 2
   * ```
   */
  div(other: ExprInput): Expr {
    return divFn(this.data, toExprData(other));
  }

  /**
   * Matrix multiplication with another expression.
   *
   * @example
   * ```ts
   * A.matmul(x)    // A @ x
   * ```
   */
  matmul(other: ExprInput): Expr {
    return matmulFn(this.data, toExprData(other));
  }

  /**
   * Dot product with another vector.
   *
   * @example
   * ```ts
   * x.dot(y)       // x' * y
   * ```
   */
  dot(other: ExprInput): Expr {
    return dotFn(this.data, toExprData(other));
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
  sum(axis?: number): Expr {
    return sumFn(this.data, axis);
  }

  /**
   * Cumulative sum of elements.
   *
   * @example
   * ```ts
   * x.cumsum()     // [x1, x1+x2, x1+x2+x3, ...]
   * ```
   */
  cumsum(axis?: number): Expr {
    return cumsumFn(this.data, axis);
  }

  /**
   * Trace of a square matrix.
   *
   * @example
   * ```ts
   * A.trace()      // tr(A)
   * ```
   */
  trace(): Expr {
    return traceFn(this.data);
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
  reshape(shape: readonly [number] | readonly [number, number]): Expr {
    return reshapeFn(this.data, shape);
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
  index(...indices: (number | readonly [number, number] | 'all')[]): Expr {
    return indexFn(this.data, ...indices);
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
  T(): Expr {
    return transposeFn(this.data);
  }

  /**
   * Transpose this matrix (alias for T()).
   */
  transpose(): Expr {
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
  diag(): Expr {
    return diagFn(this.data);
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
  norm1(): Expr {
    return norm1Fn(this.data);
  }

  /**
   * L2 norm (Euclidean norm).
   *
   * @example
   * ```ts
   * x.norm2()      // ||x||_2 = sqrt(sum(x_i^2))
   * ```
   */
  norm2(): Expr {
    return norm2Fn(this.data);
  }

  /**
   * Infinity norm (maximum absolute value).
   *
   * @example
   * ```ts
   * x.normInf()    // ||x||_inf = max(|x_i|)
   * ```
   */
  normInf(): Expr {
    return normInfFn(this.data);
  }

  /**
   * Sum of squares.
   *
   * @example
   * ```ts
   * x.sumSquares() // ||x||_2^2 = sum(x_i^2)
   * ```
   */
  sumSquares(): Expr {
    return sumSquaresFn(this.data);
  }

  /**
   * Quadratic form: x' P x.
   *
   * @example
   * ```ts
   * x.quadForm(P)  // x' * P * x
   * ```
   */
  quadForm(P: ExprInput): Expr {
    return quadFormFn(this.data, toExprData(P));
  }

  /**
   * Quadratic over linear: ||x||^2 / y.
   *
   * @example
   * ```ts
   * x.quadOverLin(y)  // ||x||^2 / y
   * ```
   */
  quadOverLin(y: ExprInput): Expr {
    return quadOverLinFn(this.data, toExprData(y));
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
  abs(): Expr {
    return absFn(this.data);
  }

  /**
   * Positive part: max(x, 0).
   *
   * @example
   * ```ts
   * x.pos()        // max(x, 0)
   * ```
   */
  pos(): Expr {
    return posFn(this.data);
  }

  /**
   * Negative part: max(-x, 0).
   *
   * @example
   * ```ts
   * x.negPart()    // max(-x, 0)
   * ```
   */
  negPart(): Expr {
    return negPartFn(this.data);
  }

  /**
   * Element-wise exponential: e^x.
   *
   * @example
   * ```ts
   * x.exp()        // e^x
   * ```
   */
  exp(): Expr {
    return expFn(this.data);
  }

  /**
   * Element-wise natural logarithm: log(x).
   *
   * @example
   * ```ts
   * x.log()        // ln(x)
   * ```
   */
  log(): Expr {
    return logFn(this.data);
  }

  /**
   * Element-wise entropy: -x * log(x).
   *
   * @example
   * ```ts
   * x.entropy()    // -x * log(x)
   * ```
   */
  entropy(): Expr {
    return entropyFn(this.data);
  }

  /**
   * Element-wise square root: sqrt(x).
   *
   * @example
   * ```ts
   * x.sqrt()       // sqrt(x)
   * ```
   */
  sqrt(): Expr {
    return sqrtFn(this.data);
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
  power(p: number): Expr {
    return powerFn(this.data, p);
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
    return eqFn(this.data, toExprData(other));
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
    return leFn(this.data, toExprData(other));
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
    return geFn(this.data, toExprData(other));
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
    return socFn(this.data, toExprData(t));
  }

  // ==================== Evaluation ====================

  /**
   * Evaluate this expression given variable values.
   *
   * @example
   * ```ts
   * const x = variable(3);
   * const expr = x.sum().mul(2);
   * const solution = await Problem.minimize(x.sum()).solve();
   *
   * // Evaluate expression at the solution
   * const result = expr.evaluate(solution.primal!);
   * console.log('2 * sum(x) =', result);
   * ```
   */
  evaluate(values: VariableValues): Float64Array {
    return evaluateFn(this.data, values);
  }

  /**
   * Evaluate this expression as a scalar given variable values.
   * Throws if the result is not a scalar.
   *
   * @example
   * ```ts
   * const x = variable(3);
   * const totalCost = x.sum();
   * const solution = await Problem.minimize(totalCost).solve();
   *
   * const cost = totalCost.value(solution.primal!);
   * console.log('Total cost:', cost);
   * ```
   */
  value(values: VariableValues): number {
    return evaluateScalarFn(this.data, values);
  }

  // ==================== Static Methods for Combining Expressions ====================

  /**
   * Vertically stack expressions.
   *
   * @example
   * ```ts
   * Expr.vstack(x, y, z)
   * ```
   */
  static vstack(...args: ExprInput[]): Expr {
    return vstackFn(...args.map(toExprData));
  }

  /**
   * Horizontally stack expressions.
   *
   * @example
   * ```ts
   * Expr.hstack(x, y, z)
   * ```
   */
  static hstack(...args: ExprInput[]): Expr {
    return hstackFn(...args.map(toExprData));
  }

  /**
   * Element-wise maximum of expressions.
   *
   * @example
   * ```ts
   * Expr.maximum(x, y, z)
   * ```
   */
  static maximum(...args: ExprInput[]): Expr {
    return maximumFn(...args.map(toExprData));
  }

  /**
   * Element-wise minimum of expressions.
   *
   * @example
   * ```ts
   * Expr.minimum(x, y, z)
   * ```
   */
  static minimum(...args: ExprInput[]): Expr {
    return minimumFn(...args.map(toExprData));
  }

  /**
   * Create an Expr from a constant value.
   *
   * @example
   * ```ts
   * Expr.constant(5)
   * Expr.constant([1, 2, 3])
   * Expr.constant([[1, 2], [3, 4]])
   * ```
   */
  static constant(value: ArrayInput): Expr {
    return new Expr(arrayToExprData(value));
  }
}

/**
 * Wrap ExprData in an Expr for fluent API access.
 * This is a convenience function for interop with existing code.
 *
 * @example
 * ```ts
 * const x = variable(5);           // Already returns Expr
 * const wrapped = wrap(someData);  // Wrap existing ExprData
 * ```
 */
export function wrap(data: ExprData): Expr {
  return new Expr(data);
}

/**
 * Check if a value is an Expr instance.
 */
export function isExpr(value: unknown): value is Expr {
  return value instanceof Expr;
}
