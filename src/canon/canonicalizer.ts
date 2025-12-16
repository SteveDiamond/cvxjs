import { ExprData, ExprId, exprShape, newExprId, ArrayData, IndexRange, Expr } from '../expr/index.js';
import { size, isScalar } from '../expr/shape.js';
import { Constraint } from '../constraints/index.js';
import {
  LinExpr,
  linExprVariable,
  linExprConstant,
  linExprAdd,
  linExprSub,
  linExprNeg,
  linExprScale,
  linExprDiagScale,
  linExprLeftMul,
  linExprSum,
  linExprVstack,
} from './lin-expr.js';
import {
  QuadExpr,
  quadExprFromLinear,
  quadExprQuadratic,
  quadExprSumSquares,
  quadExprAdd,
  quadExprScale,
} from './quad-expr.js';
import { ConeConstraint } from './cone-constraint.js';
import {
  CscMatrix,
  cscFromTriplets,
  cscFromDense,
  cscTranspose,
  cscMulMatTransposeLeft,
} from '../sparse/index.js';
import { DcpError } from '../error.js';
import { curvature, Curvature } from '../dcp/index.js';

/**
 * Canonical expression: either linear or quadratic.
 * Quadratic is only used for objectives, not constraints.
 */
export type CanonExpr =
  | { readonly kind: 'linear'; readonly expr: LinExpr }
  | { readonly kind: 'quadratic'; readonly expr: QuadExpr };

/**
 * Get the linear expression from a CanonExpr.
 * Throws if quadratic.
 */
export function canonExprAsLinear(ce: CanonExpr): LinExpr {
  if (ce.kind === 'quadratic') {
    throw new DcpError('Expected linear expression, got quadratic');
  }
  return ce.expr;
}

/**
 * Convert a CanonExpr to QuadExpr (linear expressions are wrapped).
 */
export function canonExprToQuadratic(ce: CanonExpr): QuadExpr {
  if (ce.kind === 'quadratic') {
    return ce.expr;
  }
  return quadExprFromLinear(ce.expr);
}

/**
 * Auxiliary variable info.
 */
export interface AuxVar {
  readonly id: ExprId;
  readonly size: number;
  readonly nonneg?: boolean;
}

/**
 * Result of canonicalizing an expression.
 */
export interface CanonResult {
  /** Canonical linear expression (for non-quadratic objectives) */
  readonly linExpr: LinExpr;
  /** Canonical quadratic expression (for QP objectives, optional) */
  readonly quadExpr?: QuadExpr;
  /** Cone constraints generated */
  readonly constraints: ConeConstraint[];
  /** Auxiliary variables introduced */
  readonly auxVars: AuxVar[];
}

/**
 * Canonicalization context.
 */
export class Canonicalizer {
  private constraints: ConeConstraint[] = [];
  private auxVars: AuxVar[] = [];

  /**
   * Create a new auxiliary variable.
   */
  private newAuxVar(size: number, nonneg = false): { id: ExprId; linExpr: LinExpr } {
    const id = newExprId();
    this.auxVars.push({ id, size, nonneg });
    return { id, linExpr: linExprVariable(id, size) };
  }

  /**
   * Create a non-negative auxiliary variable.
   */
  private newNonnegAuxVar(size: number): { id: ExprId; linExpr: LinExpr } {
    return this.newAuxVar(size, true);
  }

  /**
   * Add a cone constraint.
   */
  private addConstraint(constraint: ConeConstraint): void {
    this.constraints.push(constraint);
  }

  /**
   * Canonicalize an expression to linear form plus cone constraints.
   *
   * @param expr - Expression to canonicalize
   * @returns Canonical linear expression
   */
  canonicalize(input: ExprData | Expr): LinExpr {
    const expr = input instanceof Expr ? input.data : input;
    switch (expr.kind) {
      // === Leaf nodes ===
      case 'variable':
        return linExprVariable(expr.id, size(expr.shape));

      case 'constant':
        return linExprConstant(this.arrayDataToFloat64(expr.value));

      // === Affine atoms ===
      case 'add':
        return linExprAdd(this.canonicalize(expr.left), this.canonicalize(expr.right));

      case 'neg':
        return linExprNeg(this.canonicalize(expr.arg));

      case 'mul': {
        // One side must be constant
        const leftCurv = curvature(expr.left);
        const rightCurv = curvature(expr.right);

        if (leftCurv === Curvature.Constant) {
          const constShape = exprShape(expr.left);
          if (isScalar(constShape)) {
            const scalar = this.getScalarConstant(expr.left);
            return linExprScale(this.canonicalize(expr.right), scalar);
          } else {
            // Element-wise multiplication with vector/matrix constant
            const diag = this.arrayDataToFloat64(
              expr.left.kind === 'constant' ? expr.left.value : { type: 'scalar', value: 0 }
            );
            return linExprDiagScale(this.canonicalize(expr.right), diag);
          }
        }
        if (rightCurv === Curvature.Constant) {
          const constShape = exprShape(expr.right);
          if (isScalar(constShape)) {
            const scalar = this.getScalarConstant(expr.right);
            return linExprScale(this.canonicalize(expr.left), scalar);
          } else {
            // Element-wise multiplication with vector/matrix constant
            const diag = this.arrayDataToFloat64(
              expr.right.kind === 'constant' ? expr.right.value : { type: 'scalar', value: 0 }
            );
            return linExprDiagScale(this.canonicalize(expr.left), diag);
          }
        }
        throw new DcpError('mul requires at least one constant operand');
      }

      case 'div': {
        // Divisor must be constant scalar
        const scalar = this.getScalarConstant(expr.right);
        if (scalar === 0) throw new DcpError('Division by zero');
        return linExprScale(this.canonicalize(expr.left), 1 / scalar);
      }

      case 'matmul': {
        // At least one operand must be constant
        const leftCurv = curvature(expr.left);
        const rightCurv = curvature(expr.right);

        if (leftCurv === Curvature.Constant) {
          // M @ x where M is constant
          const M = this.exprToCscMatrix(expr.left);
          return linExprLeftMul(M, this.canonicalize(expr.right));
        }
        if (rightCurv === Curvature.Constant) {
          // x @ M where M is constant
          // x @ M = (M' @ x')'
          // For vectors: result = M' @ x
          const M = this.exprToCscMatrix(expr.right);
          const Mt = cscTranspose(M);
          return linExprLeftMul(Mt, this.canonicalize(expr.left));
        }
        throw new DcpError('matmul requires at least one constant operand');
      }

      case 'sum': {
        const argLin = this.canonicalize(expr.arg);

        if (expr.axis === undefined) {
          // Sum all elements
          return linExprSum(argLin);
        }

        // Sum along a specific axis
        const argShape = exprShape(expr.arg);
        if (argShape.dims.length !== 2) {
          throw new DcpError('sum with axis only supported for 2D arrays');
        }

        const m = argShape.dims[0]!;
        const n = argShape.dims[1]!;

        // Build summation matrix
        // For axis=0 (sum columns): result is (1 x n), each element is sum of column
        // For axis=1 (sum rows): result is (m x 1), each element is sum of row
        const rows: number[] = [];
        const cols: number[] = [];
        const vals: number[] = [];

        if (expr.axis === 0) {
          // Sum columns: output[j] = sum_i(input[i, j])
          // In column-major: output[j] = sum of elements at positions j*m to j*m+m-1
          for (let j = 0; j < n; j++) {
            for (let i = 0; i < m; i++) {
              rows.push(j);
              cols.push(j * m + i);
              vals.push(1);
            }
          }
          const sumMatrix = cscFromTriplets(n, m * n, rows, cols, vals);
          return linExprLeftMul(sumMatrix, argLin);
        } else {
          // Sum rows: output[i] = sum_j(input[i, j])
          // In column-major: output[i] = sum_j(input[j*m + i])
          for (let i = 0; i < m; i++) {
            for (let j = 0; j < n; j++) {
              rows.push(i);
              cols.push(j * m + i);
              vals.push(1);
            }
          }
          const sumMatrix = cscFromTriplets(m, m * n, rows, cols, vals);
          return linExprLeftMul(sumMatrix, argLin);
        }
      }

      case 'reshape': {
        // Reshape is just reinterpretation, no change to linear form
        return this.canonicalize(expr.arg);
      }

      case 'index': {
        // Indexing extracts elements from the coefficient matrices using a selection matrix
        const argLin = this.canonicalize(expr.arg);
        const argShape = exprShape(expr.arg);
        const selectionMatrix = this.buildSelectionMatrix(argShape, expr.ranges);
        return linExprLeftMul(selectionMatrix, argLin);
      }

      case 'vstack': {
        const canonArgs = expr.args.map((arg) => this.canonicalize(arg));
        return linExprVstack(canonArgs);
      }

      case 'hstack': {
        // In column-major flattened form, hstack is equivalent to vstack
        // For hstack([A, B]) where A is m x n1 and B is m x n2:
        // Result is m x (n1+n2), and flattened = [A_flat, B_flat]
        const canonArgs = expr.args.map((arg) => this.canonicalize(arg));
        return linExprVstack(canonArgs);
      }

      case 'transpose': {
        // For vectors, transpose is no-op
        const argShape = exprShape(expr.arg);
        if (argShape.dims.length <= 1) {
          return this.canonicalize(expr.arg);
        }
        throw new DcpError('transpose of matrices not yet supported');
      }

      case 'trace': {
        // trace(M) = sum of diagonal elements
        // Build selection matrix that extracts diagonal elements
        const argLin = this.canonicalize(expr.arg);
        const argShape = exprShape(expr.arg);

        if (argShape.dims.length !== 2) {
          throw new DcpError('trace requires 2D matrix argument');
        }

        const m = argShape.dims[0]!;
        const n = argShape.dims[1]!;
        const minDim = Math.min(m, n);

        // Selection vector extracts diagonal elements and sums them
        // Diagonal position (i, i) in column-major order is at flat index i * m + i
        const rows: number[] = [];
        const cols: number[] = [];
        const vals: number[] = [];

        for (let i = 0; i < minDim; i++) {
          rows.push(0);
          cols.push(i * m + i);
          vals.push(1);
        }

        const selectMatrix = cscFromTriplets(1, m * n, rows, cols, vals);
        return linExprLeftMul(selectMatrix, argLin);
      }

      case 'diag': {
        const argLin = this.canonicalize(expr.arg);
        const argShape = exprShape(expr.arg);

        if (argShape.dims.length === 1) {
          // Vector -> diagonal matrix
          // Input: n-vector, Output: n×n matrix (n² elements, column-major)
          // Place input[i] at position (i, i) = i * n + i
          const n = argShape.dims[0]!;

          const rows: number[] = [];
          const cols: number[] = [];
          const vals: number[] = [];

          for (let i = 0; i < n; i++) {
            // Output position: (i, i) in column-major = i * n + i
            rows.push(i * n + i);
            // Input position: i
            cols.push(i);
            vals.push(1);
          }

          const expandMatrix = cscFromTriplets(n * n, n, rows, cols, vals);
          return linExprLeftMul(expandMatrix, argLin);
        } else {
          // Matrix -> diagonal vector
          // Input: m×n matrix (m*n elements), Output: min(m,n)-vector
          // Extract elements at positions (i, i) = i * m + i
          const m = argShape.dims[0]!;
          const n = argShape.dims[1]!;
          const minDim = Math.min(m, n);

          const rows: number[] = [];
          const cols: number[] = [];
          const vals: number[] = [];

          for (let i = 0; i < minDim; i++) {
            // Output position: i
            rows.push(i);
            // Input position: (i, i) in column-major = i * m + i
            cols.push(i * m + i);
            vals.push(1);
          }

          const selectMatrix = cscFromTriplets(minDim, m * n, rows, cols, vals);
          return linExprLeftMul(selectMatrix, argLin);
        }
      }

      case 'cumsum': {
        // Cumulative sum: build lower-triangular summation matrix
        // For vector x = [x1, x2, x3], cumsum(x) = [x1, x1+x2, x1+x2+x3]
        // L @ x where L is lower-triangular matrix of ones
        const argLin = this.canonicalize(expr.arg);
        const argShape = exprShape(expr.arg);

        if (expr.axis === undefined) {
          // Cumsum of flattened array (1D)
          const n = argLin.rows;

          const rows: number[] = [];
          const cols: number[] = [];
          const vals: number[] = [];

          for (let i = 0; i < n; i++) {
            for (let j = 0; j <= i; j++) {
              rows.push(i);
              cols.push(j);
              vals.push(1);
            }
          }

          const cumsumMatrix = cscFromTriplets(n, n, rows, cols, vals);
          return linExprLeftMul(cumsumMatrix, argLin);
        } else {
          // Cumsum along axis for 2D array
          if (argShape.dims.length !== 2) {
            throw new DcpError('cumsum with axis only supported for 2D arrays');
          }

          const m = argShape.dims[0]!;
          const n = argShape.dims[1]!;

          const rows: number[] = [];
          const cols: number[] = [];
          const vals: number[] = [];

          if (expr.axis === 0) {
            // Cumsum along columns: for each column, apply cumsum
            // In column-major: column j starts at j*m, elements are j*m, j*m+1, ..., j*m+m-1
            for (let j = 0; j < n; j++) {
              for (let i = 0; i < m; i++) {
                for (let k = 0; k <= i; k++) {
                  rows.push(j * m + i); // output position
                  cols.push(j * m + k); // input position
                  vals.push(1);
                }
              }
            }
          } else {
            // Cumsum along rows: for each row, apply cumsum across columns
            // In column-major: row i consists of elements i, m+i, 2m+i, ...
            for (let i = 0; i < m; i++) {
              for (let j = 0; j < n; j++) {
                for (let k = 0; k <= j; k++) {
                  rows.push(j * m + i); // output position
                  cols.push(k * m + i); // input position
                  vals.push(1);
                }
              }
            }
          }

          const cumsumMatrix = cscFromTriplets(m * n, m * n, rows, cols, vals);
          return linExprLeftMul(cumsumMatrix, argLin);
        }
      }

      // === Nonlinear convex atoms ===
      case 'norm1': {
        // ||x||_1 = sum(|x_i|)
        // Introduce t_i >= |x_i|, which means t_i >= x_i and t_i >= -x_i
        const x = this.canonicalize(expr.arg);
        const n = x.rows;

        const { linExpr: t } = this.newNonnegAuxVar(n);

        // t >= x  =>  t - x >= 0
        this.addConstraint({ kind: 'nonneg', a: linExprSub(t, x) });
        // t >= -x  =>  t + x >= 0
        this.addConstraint({ kind: 'nonneg', a: linExprAdd(t, x) });

        return linExprSum(t);
      }

      case 'norm2': {
        // ||x||_2: introduce t >= 0, (t, x) in SOC
        const x = this.canonicalize(expr.arg);
        const { linExpr: t } = this.newNonnegAuxVar(1);

        this.addConstraint({ kind: 'soc', t, x });

        return t;
      }

      case 'normInf': {
        // ||x||_inf = max(|x_i|)
        // Introduce t >= |x_i| for all i
        const x = this.canonicalize(expr.arg);
        const n = x.rows;

        const { linExpr: t } = this.newNonnegAuxVar(1);

        // Broadcast t to size n
        const tBroadcast = this.broadcastScalar(t, n);

        // t >= x  =>  t - x >= 0
        this.addConstraint({ kind: 'nonneg', a: linExprSub(tBroadcast, x) });
        // t >= -x  =>  t + x >= 0
        this.addConstraint({ kind: 'nonneg', a: linExprAdd(tBroadcast, x) });

        return t;
      }

      case 'abs': {
        // |x|: introduce t >= |x|, return t
        const x = this.canonicalize(expr.arg);
        const n = x.rows;

        const { linExpr: t } = this.newNonnegAuxVar(n);

        // t >= x and t >= -x
        this.addConstraint({ kind: 'nonneg', a: linExprSub(t, x) });
        this.addConstraint({ kind: 'nonneg', a: linExprAdd(t, x) });

        return t;
      }

      case 'pos': {
        // max(x, 0): introduce t >= 0, t >= x, return t
        const x = this.canonicalize(expr.arg);
        const n = x.rows;

        const { linExpr: t } = this.newNonnegAuxVar(n);

        // t >= x
        this.addConstraint({ kind: 'nonneg', a: linExprSub(t, x) });

        return t;
      }

      case 'negPart': {
        // max(-x, 0): introduce t >= 0, t >= -x, return t
        // This is equivalent to pos(-x)
        const x = this.canonicalize(expr.arg);
        const negX = linExprNeg(x);
        const n = x.rows;

        const { linExpr: t } = this.newNonnegAuxVar(n);

        // t >= -x
        this.addConstraint({ kind: 'nonneg', a: linExprSub(t, negX) });

        return t;
      }

      case 'maximum': {
        // max(x_1, ..., x_k): introduce t >= x_i for all i
        const args = expr.args.map((arg) => this.canonicalize(arg));
        const n = args[0]!.rows;

        const { linExpr: t } = this.newAuxVar(n);

        for (const arg of args) {
          this.addConstraint({ kind: 'nonneg', a: linExprSub(t, arg) });
        }

        return t;
      }

      case 'sumSquares': {
        // ||x||_2^2 = x' x
        // For objectives, we use native QP via QuadExpr
        // For constraints, use SOC reformulation

        const x = this.canonicalize(expr.arg);

        // SOC reformulation: introduce s where ||x|| <= s
        const { linExpr: s } = this.newNonnegAuxVar(1);
        this.addConstraint({ kind: 'soc', t: s, x });

        // Return s (the bound on ||x||)
        // For objectives, canonicalizeObjective will handle creating QuadExpr
        return s;
      }

      case 'quadForm': {
        // x' P x where P is constant PSD
        // For objectives, we use native QP via QuadExpr
        // For constraints, use SOC reformulation via Cholesky

        const x = this.canonicalize(expr.x);

        // SOC reformulation: ||L'x|| <= t where P = LL'
        // For now, treat as ||x|| scaled
        const { linExpr: s } = this.newNonnegAuxVar(1);
        this.addConstraint({ kind: 'soc', t: s, x });

        // Return s as approximation
        return s;
      }

      case 'quadOverLin': {
        // ||x||^2 / y: hyperbolic constraint
        throw new DcpError('quadOverLin canonicalization not yet supported');
      }

      case 'exp': {
        // exp(x): introduce t >= exp(x)
        // Using exp cone: (x, 1, t) means 1 * exp(x/1) <= t, i.e., exp(x) <= t
        const x = this.canonicalize(expr.arg);
        const n = x.rows;

        // For each element, introduce t_i and exp cone constraint
        const { linExpr: t } = this.newNonnegAuxVar(n);

        // Create scalar 1 for y component
        const one = linExprConstant(new Float64Array([1]));

        // For element-wise exp, we need n exp cone constraints
        for (let i = 0; i < n; i++) {
          // Extract element i of x
          const selectRows: number[] = [0];
          const selectCols: number[] = [i];
          const selectVals: number[] = [1];
          const selectMatrix = cscFromTriplets(1, n, selectRows, selectCols, selectVals);

          const xi = linExprLeftMul(selectMatrix, x);
          const ti = linExprLeftMul(selectMatrix, t);

          this.addConstraint({ kind: 'exp', x: xi, y: one, z: ti });
        }

        return t;
      }

      // === Nonlinear concave atoms ===
      case 'log': {
        // log(x): introduce t <= log(x)
        // Using exp cone: (t, 1, x) means 1 * exp(t/1) <= x, i.e., exp(t) <= x, i.e., t <= log(x)
        const x = this.canonicalize(expr.arg);
        const n = x.rows;

        // For each element, introduce t_i and exp cone constraint
        const { linExpr: t } = this.newAuxVar(n);

        // Create scalar 1 for y component
        const one = linExprConstant(new Float64Array([1]));

        // For element-wise log, we need n exp cone constraints
        for (let i = 0; i < n; i++) {
          // Extract element i
          const selectRows: number[] = [0];
          const selectCols: number[] = [i];
          const selectVals: number[] = [1];
          const selectMatrix = cscFromTriplets(1, n, selectRows, selectCols, selectVals);

          const xi = linExprLeftMul(selectMatrix, x);
          const ti = linExprLeftMul(selectMatrix, t);

          // (t, 1, x) in ExpCone
          this.addConstraint({ kind: 'exp', x: ti, y: one, z: xi });
        }

        return t;
      }

      case 'entropy': {
        // -x*log(x): introduce t <= -x*log(x)
        // Using exp cone: (t, x, 1) means x * exp(t/x) <= 1
        // => exp(t/x) <= 1/x => t/x <= log(1/x) = -log(x) => t <= -x*log(x)
        const x = this.canonicalize(expr.arg);
        const n = x.rows;

        // For each element, introduce t_i and exp cone constraint
        const { linExpr: t } = this.newAuxVar(n);

        // Create scalar 1 for z component
        const one = linExprConstant(new Float64Array([1]));

        // For element-wise entropy, we need n exp cone constraints
        for (let i = 0; i < n; i++) {
          // Extract element i
          const selectRows: number[] = [0];
          const selectCols: number[] = [i];
          const selectVals: number[] = [1];
          const selectMatrix = cscFromTriplets(1, n, selectRows, selectCols, selectVals);

          const xi = linExprLeftMul(selectMatrix, x);
          const ti = linExprLeftMul(selectMatrix, t);

          // (t, x, 1) in ExpCone
          this.addConstraint({ kind: 'exp', x: ti, y: xi, z: one });
        }

        return t;
      }

      case 'sqrt': {
        // sqrt(x): introduce t <= sqrt(x) = x^0.5
        // Using power cone with alpha=0.5: x^0.5 * 1^0.5 >= |z|
        // (x, 1, t) in PowerCone(0.5) gives us sqrt(x) >= t
        const x = this.canonicalize(expr.arg);
        const n = x.rows;

        const { linExpr: t } = this.newNonnegAuxVar(n);
        const one = linExprConstant(new Float64Array([1]));

        for (let i = 0; i < n; i++) {
          const selectRows: number[] = [0];
          const selectCols: number[] = [i];
          const selectVals: number[] = [1];
          const selectMatrix = cscFromTriplets(1, n, selectRows, selectCols, selectVals);

          const xi = linExprLeftMul(selectMatrix, x);
          const ti = linExprLeftMul(selectMatrix, t);

          // (x, 1, t) in PowerCone(0.5)
          this.addConstraint({ kind: 'power', x: xi, y: one, z: ti, alpha: 0.5 });
        }

        return t;
      }

      case 'power': {
        // x^p: curvature depends on p
        const x = this.canonicalize(expr.arg);
        const n = x.rows;
        const p = expr.p;

        const { linExpr: t } = this.newNonnegAuxVar(n);
        const one = linExprConstant(new Float64Array([1]));

        if (p > 0 && p < 1) {
          // Concave case: t <= x^p
          // (x, 1, t) in PowerCone(p) gives us x^p >= t
          for (let i = 0; i < n; i++) {
            const selectRows: number[] = [0];
            const selectCols: number[] = [i];
            const selectVals: number[] = [1];
            const selectMatrix = cscFromTriplets(1, n, selectRows, selectCols, selectVals);

            const xi = linExprLeftMul(selectMatrix, x);
            const ti = linExprLeftMul(selectMatrix, t);

            this.addConstraint({ kind: 'power', x: xi, y: one, z: ti, alpha: p });
          }
        } else if (p > 1) {
          // Convex case: t >= x^p
          // Equivalent to x <= t^(1/p), i.e., (t, 1, x) in PowerCone(1/p)
          for (let i = 0; i < n; i++) {
            const selectRows: number[] = [0];
            const selectCols: number[] = [i];
            const selectVals: number[] = [1];
            const selectMatrix = cscFromTriplets(1, n, selectRows, selectCols, selectVals);

            const xi = linExprLeftMul(selectMatrix, x);
            const ti = linExprLeftMul(selectMatrix, t);

            this.addConstraint({ kind: 'power', x: ti, y: one, z: xi, alpha: 1 / p });
          }
        } else if (p < 0) {
          // x^p for p < 0 is convex but more complex to handle
          throw new DcpError(`power with p=${p} < 0 not yet supported`);
        }

        return t;
      }

      case 'minimum': {
        // min(x_1, ..., x_k): introduce t <= x_i for all i
        const args = expr.args.map((arg) => this.canonicalize(arg));
        const n = args[0]!.rows;

        const { linExpr: t } = this.newAuxVar(n);

        for (const arg of args) {
          // t <= arg  =>  arg - t >= 0
          this.addConstraint({ kind: 'nonneg', a: linExprSub(arg, t) });
        }

        return t;
      }

      default: {
        const _exhaustive: never = expr;
        throw new DcpError(`Unknown expression kind: ${(expr as ExprData).kind}`);
      }
    }
  }

  /**
   * Canonicalize constraints.
   */
  canonicalizeConstraint(constraint: Constraint): void {
    switch (constraint.kind) {
      case 'zero': {
        const a = this.canonicalize(constraint.expr);
        this.addConstraint({ kind: 'zero', a });
        break;
      }

      case 'nonneg': {
        const a = this.canonicalize(constraint.expr);
        this.addConstraint({ kind: 'nonneg', a });
        break;
      }

      case 'soc': {
        const t = this.canonicalize(constraint.t);
        const x = this.canonicalize(constraint.x);
        this.addConstraint({ kind: 'soc', t, x });
        break;
      }
    }
  }

  /**
   * Canonicalize an objective expression, potentially returning a quadratic expression.
   *
   * For quadratic objectives (sumSquares, quadForm), this method returns a QuadExpr
   * that can be directly converted to the P matrix for Clarabel's native QP support.
   *
   * @param expr - Expression to canonicalize
   * @returns Canonical expression (linear or quadratic)
   */
  canonicalizeObjective(input: ExprData | Expr): CanonExpr {
    const expr = input instanceof Expr ? input.data : input;
    // Check for quadratic objective patterns
    if (expr.kind === 'sumSquares') {
      // ||x||^2 = x' I x - use native QP
      const argExpr = expr.arg;

      // If the argument is a simple variable, use native QP
      if (argExpr.kind === 'variable') {
        const varId = argExpr.id;
        const varSize = size(argExpr.shape);
        return {
          kind: 'quadratic',
          expr: quadExprSumSquares(varId, varSize),
        };
      }

      // If the argument is affine (Ax + b), we can still use QP
      // ||Ax + b||^2 = (Ax + b)' (Ax + b) = x' A'A x + 2b'Ax + b'b
      // For now, check if arg canonicalizes to a simple form
      const argLin = this.canonicalize(argExpr);
      const vars = Array.from(argLin.coeffs.keys());

      // Check if constant term is zero and there's exactly one variable
      const hasConstant = argLin.constant.some((c) => c !== 0);
      if (vars.length === 1 && !hasConstant) {
        const varId = vars[0]!;
        const A = argLin.coeffs.get(varId)!;
        // P = A' A (for ||Ax||^2 = x' A'A x)
        const AtA = cscMulMatTransposeLeft(A, A);
        return {
          kind: 'quadratic',
          expr: quadExprQuadratic(varId, AtA),
        };
      }

      // Fall back to SOC reformulation for complex cases
    }

    if (expr.kind === 'quadForm') {
      // x' P x - use native QP
      const xExpr = expr.x;
      const pExpr = expr.P;

      // If x is a simple variable and P is constant, use native QP
      if (xExpr.kind === 'variable' && pExpr.kind === 'constant') {
        const varId = xExpr.id;
        const P = this.exprToCscMatrix(pExpr);
        return {
          kind: 'quadratic',
          expr: quadExprQuadratic(varId, P),
        };
      }

      // Fall back to SOC reformulation for complex cases
    }

    // For add expressions, check if both sides have quadratic parts
    if (expr.kind === 'add') {
      const leftCanon = this.canonicalizeObjective(expr.left);
      const rightCanon = this.canonicalizeObjective(expr.right);

      // If either is quadratic, combine as quadratic
      if (leftCanon.kind === 'quadratic' || rightCanon.kind === 'quadratic') {
        const leftQuad = canonExprToQuadratic(leftCanon);
        const rightQuad = canonExprToQuadratic(rightCanon);
        return {
          kind: 'quadratic',
          expr: quadExprAdd(leftQuad, rightQuad),
        };
      }

      // Both linear - return linear
      return {
        kind: 'linear',
        expr: linExprAdd(leftCanon.expr, rightCanon.expr),
      };
    }

    // Default: use regular canonicalization and wrap as linear
    return {
      kind: 'linear',
      expr: this.canonicalize(expr),
    };
  }

  /**
   * Get the collected constraints.
   */
  getConstraints(): ConeConstraint[] {
    return this.constraints;
  }

  /**
   * Get the collected auxiliary variables.
   */
  getAuxVars(): AuxVar[] {
    return this.auxVars;
  }

  /**
   * Helper to convert ArrayData to Float64Array.
   */
  private arrayDataToFloat64(data: ArrayData): Float64Array {
    switch (data.type) {
      case 'scalar':
        return new Float64Array([data.value]);
      case 'dense':
        return data.data;
      case 'sparse': {
        const result = new Float64Array(size(data.shape));
        for (let c = 0; c < data.shape.dims[1]!; c++) {
          for (let i = data.colPtr[c]!; i < data.colPtr[c + 1]!; i++) {
            result[c * data.shape.dims[0]! + data.rowIdx[i]!] = data.values[i]!;
          }
        }
        return result;
      }
    }
  }

  /**
   * Get scalar constant value from expression.
   */
  private getScalarConstant(input: ExprData | Expr): number {
    const expr = input instanceof Expr ? input.data : input;
    if (expr.kind !== 'constant') {
      throw new DcpError('Expected constant expression');
    }
    if (expr.value.type === 'scalar') {
      return expr.value.value;
    }
    const arr = this.arrayDataToFloat64(expr.value);
    if (arr.length !== 1) {
      throw new DcpError('Expected scalar constant');
    }
    return arr[0]!;
  }

  /**
   * Convert expression to CSC matrix (must be constant).
   */
  private exprToCscMatrix(input: ExprData | Expr): CscMatrix {
    const expr = input instanceof Expr ? input.data : input;
    if (expr.kind !== 'constant') {
      throw new DcpError('Expected constant expression');
    }

    const shape = exprShape(expr);
    const data = this.arrayDataToFloat64(expr.value);

    if (shape.dims.length === 0) {
      // Scalar
      return cscFromTriplets(1, 1, [0], [0], [data[0]!]);
    } else if (shape.dims.length === 1) {
      // Vector - treat as column vector
      const n = shape.dims[0]!;
      const rows: number[] = [];
      const cols: number[] = [];
      const vals: number[] = [];
      for (let i = 0; i < n; i++) {
        if (data[i] !== 0) {
          rows.push(i);
          cols.push(0);
          vals.push(data[i]!);
        }
      }
      return cscFromTriplets(n, 1, rows, cols, vals);
    } else {
      // Matrix (data is column-major)
      const nrows = shape.dims[0]!;
      const ncols = shape.dims[1]!;
      return cscFromDense(nrows, ncols, data);
    }
  }

  /**
   * Broadcast a scalar LinExpr to size n.
   */
  private broadcastScalar(scalar: LinExpr, n: number): LinExpr {
    if (scalar.rows !== 1) {
      throw new Error('Expected scalar LinExpr');
    }

    // Create broadcast matrix: all ones column
    const broadcastRows: number[] = [];
    const broadcastCols: number[] = [];
    const broadcastVals: number[] = [];
    for (let i = 0; i < n; i++) {
      broadcastRows.push(i);
      broadcastCols.push(0);
      broadcastVals.push(1);
    }
    const broadcast = cscFromTriplets(n, 1, broadcastRows, broadcastCols, broadcastVals);

    return linExprLeftMul(broadcast, scalar);
  }

  /**
   * Build a selection matrix for indexing operations.
   *
   * The selection matrix S extracts elements from a flattened (column-major)
   * array such that: result = S @ flatten(input)
   *
   * @param inputShape - Shape of the input expression
   * @param ranges - Index ranges for each dimension
   * @returns Sparse selection matrix
   */
  private buildSelectionMatrix(
    inputShape: { dims: readonly number[] },
    ranges: readonly IndexRange[]
  ): CscMatrix {
    // Normalize input shape to 2D: treat 1D as [n, 1], scalar as [1, 1]
    const dims = inputShape.dims;
    const m = dims[0] ?? 1; // rows
    const n = dims[1] ?? 1; // cols
    const inputSize = m * n;

    // Normalize ranges: pad with 'all' if fewer ranges than dimensions
    const normalizedRanges: IndexRange[] = [];
    for (let d = 0; d < 2; d++) {
      if (d < ranges.length) {
        normalizedRanges.push(ranges[d]!);
      } else {
        normalizedRanges.push({ type: 'all' });
      }
    }

    // Compute dimension ranges and track which dimensions are kept
    type DimRange = { start: number; stop: number; keep: boolean };
    const dimRanges: DimRange[] = [];
    const outputDims: number[] = [];

    for (let d = 0; d < 2; d++) {
      const range = normalizedRanges[d]!;
      const dimSize = d === 0 ? m : n;

      if (range.type === 'single') {
        // Single index: extract one element, dimension is removed
        if (range.index < 0 || range.index >= dimSize) {
          throw new DcpError(
            `Index ${range.index} out of bounds for dimension ${d} with size ${dimSize}`
          );
        }
        dimRanges.push({ start: range.index, stop: range.index + 1, keep: false });
      } else if (range.type === 'range') {
        // Range: extract elements [start, stop)
        if (range.start < 0 || range.stop > dimSize || range.start > range.stop) {
          throw new DcpError(
            `Invalid range [${range.start}, ${range.stop}) for dimension ${d} with size ${dimSize}`
          );
        }
        const rangeSize = range.stop - range.start;
        if (rangeSize > 0) {
          dimRanges.push({ start: range.start, stop: range.stop, keep: true });
          outputDims.push(rangeSize);
        } else {
          // Empty range
          dimRanges.push({ start: range.start, stop: range.stop, keep: true });
          outputDims.push(0);
        }
      } else {
        // 'all': keep entire dimension
        dimRanges.push({ start: 0, stop: dimSize, keep: true });
        outputDims.push(dimSize);
      }
    }

    // Compute output size
    const outputRows = outputDims[0] ?? 1;
    const outputCols = outputDims[1] ?? 1;
    const outputSize = outputRows * outputCols;

    // Handle empty output case
    if (outputSize === 0) {
      return cscFromTriplets(0, inputSize, [], [], []);
    }

    // Build selection matrix triplets
    // For each output element (in column-major order), find the corresponding input element
    const selRows: number[] = [];
    const selCols: number[] = [];
    const selVals: number[] = [];

    const rowRange = dimRanges[0]!;
    const colRange = dimRanges[1]!;

    let outIdx = 0;
    for (let outC = 0; outC < outputCols; outC++) {
      for (let outR = 0; outR < outputRows; outR++) {
        // Map output position to source position
        const srcR = rowRange.start + outR;
        const srcC = colRange.start + outC;
        // Source flat index (column-major)
        const srcIdx = srcC * m + srcR;

        selRows.push(outIdx);
        selCols.push(srcIdx);
        selVals.push(1);
        outIdx++;
      }
    }

    return cscFromTriplets(outputSize, inputSize, selRows, selCols, selVals);
  }
}

/**
 * Canonicalize a problem to standard form.
 */
export function canonicalizeProblem(
  objective: ExprData | Expr,
  constraints: Constraint[],
  sense: 'minimize' | 'maximize'
): {
  objectiveLinExpr: LinExpr;
  objectiveQuadExpr?: QuadExpr;
  coneConstraints: ConeConstraint[];
  auxVars: AuxVar[];
  objectiveOffset: number;
} {
  const canonicalizer = new Canonicalizer();

  // Canonicalize objective (may return quadratic for QP)
  let canonExpr = canonicalizer.canonicalizeObjective(objective);
  let objectiveOffset = 0;

  // For maximization, negate the objective
  if (sense === 'maximize') {
    if (canonExpr.kind === 'quadratic') {
      canonExpr = { kind: 'quadratic', expr: quadExprScale(canonExpr.expr, -1) };
    } else {
      canonExpr = { kind: 'linear', expr: linExprNeg(canonExpr.expr) };
    }
  }

  // Extract the linear expression and optional quadratic expression
  let objectiveLinExpr: LinExpr;
  let objectiveQuadExpr: QuadExpr | undefined;

  if (canonExpr.kind === 'quadratic') {
    objectiveQuadExpr = canonExpr.expr;
    objectiveLinExpr = canonExpr.expr.linear;
    objectiveOffset = canonExpr.expr.constant;
  } else {
    objectiveLinExpr = canonExpr.expr;
    // Extract constant offset from linear objective
    if (objectiveLinExpr.rows === 1) {
      objectiveOffset = objectiveLinExpr.constant[0]!;
      objectiveLinExpr = {
        ...objectiveLinExpr,
        constant: new Float64Array(1), // Zero out constant
      };
    }
  }

  // Canonicalize constraints
  for (const c of constraints) {
    canonicalizer.canonicalizeConstraint(c);
  }

  return {
    objectiveLinExpr,
    objectiveQuadExpr,
    coneConstraints: canonicalizer.getConstraints(),
    auxVars: canonicalizer.getAuxVars(),
    objectiveOffset,
  };
}
