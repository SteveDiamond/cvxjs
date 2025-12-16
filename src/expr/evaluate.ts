import { ExprData, ExprId, ArrayData, exprShape } from './expr-data.js';
import { size } from './shape.js';
import { Expr } from './expr.js';

/**
 * Variable values mapping - variable ID to its Float64Array values.
 */
export type VariableValues = ReadonlyMap<ExprId, Float64Array>;

/**
 * Evaluate an expression given variable values.
 *
 * @example
 * ```ts
 * const x = variable(3);
 * const expr = x.sum().mul(2);
 * const solution = await Problem.minimize(expr).solve();
 *
 * // Evaluate expression at the solution
 * const result = evaluate(expr, solution.primal!);
 * console.log('2 * sum(x) =', result);
 * ```
 */
export function evaluate(expr: ExprData | Expr, values: VariableValues): Float64Array {
  const data = expr instanceof Expr ? expr.data : expr;
  return evalExpr(data, values);
}

/**
 * Evaluate an expression and return a scalar value.
 * Throws if the result is not a scalar.
 */
export function evaluateScalar(expr: ExprData | Expr, values: VariableValues): number {
  const result = evaluate(expr, values);
  if (result.length !== 1) {
    throw new Error(`Expected scalar result, got ${result.length} elements`);
  }
  return result[0]!;
}

function evalExpr(expr: ExprData, values: VariableValues): Float64Array {
  switch (expr.kind) {
    case 'variable': {
      const val = values.get(expr.id);
      if (!val) {
        throw new Error(`Variable ${expr.id} not found in values`);
      }
      return val;
    }

    case 'constant':
      return arrayDataToFloat64(expr.value);

    case 'add': {
      const left = evalExpr(expr.left, values);
      const right = evalExpr(expr.right, values);
      return broadcast(left, right, (a, b) => a + b);
    }

    case 'neg': {
      const arg = evalExpr(expr.arg, values);
      return arg.map((v) => -v);
    }

    case 'mul': {
      const left = evalExpr(expr.left, values);
      const right = evalExpr(expr.right, values);
      return broadcast(left, right, (a, b) => a * b);
    }

    case 'div': {
      const left = evalExpr(expr.left, values);
      const right = evalExpr(expr.right, values);
      return broadcast(left, right, (a, b) => a / b);
    }

    case 'matmul': {
      const left = evalExpr(expr.left, values);
      const right = evalExpr(expr.right, values);
      const leftShape = exprShape(expr.left);
      const rightShape = exprShape(expr.right);
      return matmul(left, leftShape.dims, right, rightShape.dims);
    }

    case 'sum': {
      const arg = evalExpr(expr.arg, values);
      if (expr.axis === undefined) {
        // Sum all elements
        let total = 0;
        for (let i = 0; i < arg.length; i++) {
          total += arg[i]!;
        }
        return new Float64Array([total]);
      }
      // Sum along axis - for simplicity, only support full sum for now
      throw new Error('Axis-based sum not yet implemented in evaluate');
    }

    case 'norm1': {
      const arg = evalExpr(expr.arg, values);
      let sum = 0;
      for (let i = 0; i < arg.length; i++) {
        sum += Math.abs(arg[i]!);
      }
      return new Float64Array([sum]);
    }

    case 'norm2': {
      const arg = evalExpr(expr.arg, values);
      let sumSq = 0;
      for (let i = 0; i < arg.length; i++) {
        sumSq += arg[i]! * arg[i]!;
      }
      return new Float64Array([Math.sqrt(sumSq)]);
    }

    case 'normInf': {
      const arg = evalExpr(expr.arg, values);
      let maxAbs = 0;
      for (let i = 0; i < arg.length; i++) {
        maxAbs = Math.max(maxAbs, Math.abs(arg[i]!));
      }
      return new Float64Array([maxAbs]);
    }

    case 'sumSquares': {
      const arg = evalExpr(expr.arg, values);
      let sum = 0;
      for (let i = 0; i < arg.length; i++) {
        sum += arg[i]! * arg[i]!;
      }
      return new Float64Array([sum]);
    }

    case 'quadForm': {
      const x = evalExpr(expr.x, values);
      const P = evalExpr(expr.P, values);
      const n = x.length;
      // x' P x
      let result = 0;
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          result += x[i]! * P[j * n + i]! * x[j]!;
        }
      }
      return new Float64Array([result]);
    }

    case 'abs': {
      const arg = evalExpr(expr.arg, values);
      return arg.map((v) => Math.abs(v));
    }

    case 'pos': {
      const arg = evalExpr(expr.arg, values);
      return arg.map((v) => Math.max(v, 0));
    }

    case 'negPart': {
      const arg = evalExpr(expr.arg, values);
      return arg.map((v) => Math.max(-v, 0));
    }

    case 'exp': {
      const arg = evalExpr(expr.arg, values);
      return arg.map((v) => Math.exp(v));
    }

    case 'log': {
      const arg = evalExpr(expr.arg, values);
      return arg.map((v) => Math.log(v));
    }

    case 'entropy': {
      const arg = evalExpr(expr.arg, values);
      return arg.map((v) => (v > 0 ? -v * Math.log(v) : 0));
    }

    case 'sqrt': {
      const arg = evalExpr(expr.arg, values);
      return arg.map((v) => Math.sqrt(v));
    }

    case 'power': {
      const arg = evalExpr(expr.arg, values);
      return arg.map((v) => Math.pow(v, expr.p));
    }

    case 'reshape': {
      // Just return the same values - reshape only affects shape, not data order
      return evalExpr(expr.arg, values);
    }

    case 'transpose': {
      const arg = evalExpr(expr.arg, values);
      const shape = exprShape(expr.arg);
      if (shape.dims.length <= 1) return arg;
      const [rows, cols] = shape.dims;
      const result = new Float64Array(arg.length);
      for (let i = 0; i < rows!; i++) {
        for (let j = 0; j < cols!; j++) {
          result[i * cols! + j] = arg[j * rows! + i]!;
        }
      }
      return result;
    }

    case 'trace': {
      const arg = evalExpr(expr.arg, values);
      const shape = exprShape(expr.arg);
      const n = shape.dims[0]!;
      let trace = 0;
      for (let i = 0; i < n; i++) {
        trace += arg[i * n + i]!;
      }
      return new Float64Array([trace]);
    }

    case 'diag': {
      const arg = evalExpr(expr.arg, values);
      const shape = exprShape(expr.arg);
      if (shape.dims.length === 1) {
        // Vector -> diagonal matrix
        const n = shape.dims[0]!;
        const result = new Float64Array(n * n);
        for (let i = 0; i < n; i++) {
          result[i * n + i] = arg[i]!;
        }
        return result;
      }
      // Matrix -> diagonal vector
      const n = Math.min(shape.dims[0]!, shape.dims[1]!);
      const result = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        result[i] = arg[i * shape.dims[0]! + i]!;
      }
      return result;
    }

    case 'cumsum': {
      const arg = evalExpr(expr.arg, values);
      const result = new Float64Array(arg.length);
      let cum = 0;
      for (let i = 0; i < arg.length; i++) {
        cum += arg[i]!;
        result[i] = cum;
      }
      return result;
    }

    case 'vstack': {
      const results = expr.args.map((a) => evalExpr(a, values));
      const totalLen = results.reduce((sum, arr) => sum + arr.length, 0);
      const result = new Float64Array(totalLen);
      let offset = 0;
      for (const arr of results) {
        result.set(arr, offset);
        offset += arr.length;
      }
      return result;
    }

    case 'hstack': {
      // For simplicity, treat as concatenation
      const results = expr.args.map((a) => evalExpr(a, values));
      const totalLen = results.reduce((sum, arr) => sum + arr.length, 0);
      const result = new Float64Array(totalLen);
      let offset = 0;
      for (const arr of results) {
        result.set(arr, offset);
        offset += arr.length;
      }
      return result;
    }

    case 'maximum': {
      const results = expr.args.map((a) => evalExpr(a, values));
      const len = results[0]!.length;
      const result = new Float64Array(len);
      for (let i = 0; i < len; i++) {
        let max = -Infinity;
        for (const arr of results) {
          max = Math.max(max, arr.length === 1 ? arr[0]! : arr[i]!);
        }
        result[i] = max;
      }
      return result;
    }

    case 'minimum': {
      const results = expr.args.map((a) => evalExpr(a, values));
      const len = results[0]!.length;
      const result = new Float64Array(len);
      for (let i = 0; i < len; i++) {
        let min = Infinity;
        for (const arr of results) {
          min = Math.min(min, arr.length === 1 ? arr[0]! : arr[i]!);
        }
        result[i] = min;
      }
      return result;
    }

    case 'index': {
      // Index is complex - simplify for now
      throw new Error('Index evaluation not yet implemented');
    }

    case 'quadOverLin': {
      const x = evalExpr(expr.x, values);
      const y = evalExpr(expr.y, values);
      let sumSq = 0;
      for (let i = 0; i < x.length; i++) {
        sumSq += x[i]! * x[i]!;
      }
      return new Float64Array([sumSq / y[0]!]);
    }
  }
}

function arrayDataToFloat64(data: ArrayData): Float64Array {
  switch (data.type) {
    case 'scalar':
      return new Float64Array([data.value]);
    case 'dense':
      return data.data;
    case 'sparse': {
      const result = new Float64Array(size(data.shape));
      const { colPtr, rowIdx, values, shape } = data;
      const nCols = shape.dims[1] ?? 1;
      for (let j = 0; j < nCols; j++) {
        for (let k = colPtr[j]!; k < colPtr[j + 1]!; k++) {
          const i = rowIdx[k]!;
          result[j * shape.dims[0]! + i] = values[k]!;
        }
      }
      return result;
    }
  }
}

function broadcast(
  left: Float64Array,
  right: Float64Array,
  op: (a: number, b: number) => number
): Float64Array {
  // Handle scalar broadcasting
  if (left.length === 1 && right.length > 1) {
    const scalar = left[0]!;
    return right.map((v) => op(scalar, v));
  }
  if (right.length === 1 && left.length > 1) {
    const scalar = right[0]!;
    return left.map((v) => op(v, scalar));
  }
  // Element-wise
  if (left.length !== right.length) {
    throw new Error(`Shape mismatch: ${left.length} vs ${right.length}`);
  }
  const result = new Float64Array(left.length);
  for (let i = 0; i < left.length; i++) {
    result[i] = op(left[i]!, right[i]!);
  }
  return result;
}

function matmul(
  left: Float64Array,
  leftDims: readonly number[],
  right: Float64Array,
  rightDims: readonly number[]
): Float64Array {
  // Handle vector-vector dot product
  if (leftDims.length === 1 && rightDims.length === 1) {
    let sum = 0;
    for (let i = 0; i < left.length; i++) {
      sum += left[i]! * right[i]!;
    }
    return new Float64Array([sum]);
  }

  // Matrix-vector
  if (rightDims.length === 1) {
    const m = leftDims[0]!;
    const k = leftDims[1]!;
    const result = new Float64Array(m);
    // left is column-major: left[j * m + i] = A[i, j]
    for (let i = 0; i < m; i++) {
      let sum = 0;
      for (let j = 0; j < k; j++) {
        sum += left[j * m + i]! * right[j]!;
      }
      result[i] = sum;
    }
    return result;
  }

  // Vector-matrix
  if (leftDims.length === 1) {
    const k = leftDims[0]!;
    const n = rightDims[1]!;
    const result = new Float64Array(n);
    for (let j = 0; j < n; j++) {
      let sum = 0;
      for (let i = 0; i < k; i++) {
        sum += left[i]! * right[j * k + i]!;
      }
      result[j] = sum;
    }
    return result;
  }

  // Matrix-matrix
  const m = leftDims[0]!;
  const k = leftDims[1]!;
  const n = rightDims[1]!;
  const result = new Float64Array(m * n);
  // Result is column-major
  for (let jout = 0; jout < n; jout++) {
    for (let i = 0; i < m; i++) {
      let sum = 0;
      for (let jin = 0; jin < k; jin++) {
        sum += left[jin * m + i]! * right[jout * k + jin]!;
      }
      result[jout * m + i] = sum;
    }
  }
  return result;
}
