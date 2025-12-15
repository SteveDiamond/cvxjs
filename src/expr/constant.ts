import { Expr, ArrayData, newExprId } from './expression.js';
import { vector, matrix } from './shape.js';

/**
 * Create a constant expression from various input types.
 *
 * @example
 * ```ts
 * const c = constant(5);                    // Scalar
 * const v = constant([1, 2, 3]);            // Vector
 * const M = constant([[1, 2], [3, 4]]);     // Matrix
 * ```
 */
export function constant(
  value: number | readonly number[] | readonly (readonly number[])[] | Float64Array
): Expr {
  return {
    kind: 'constant',
    id: newExprId(),
    value: toArrayData(value),
  };
}

/**
 * Create a constant expression from pre-built ArrayData.
 */
export function constantFromData(data: ArrayData): Expr {
  return {
    kind: 'constant',
    id: newExprId(),
    value: data,
  };
}

/**
 * Convert various input types to ArrayData.
 */
export function toArrayData(
  value: number | readonly number[] | readonly (readonly number[])[] | Float64Array | ArrayData
): ArrayData {
  // Already ArrayData
  if (typeof value === 'object' && 'type' in value) {
    return value;
  }

  // Scalar
  if (typeof value === 'number') {
    return { type: 'scalar', value };
  }

  // Float64Array (dense vector)
  if (value instanceof Float64Array) {
    return {
      type: 'dense',
      data: value,
      shape: vector(value.length),
    };
  }

  // 1D array (vector)
  if (value.length === 0 || typeof value[0] === 'number') {
    const arr = value as readonly number[];
    return {
      type: 'dense',
      data: new Float64Array(arr),
      shape: vector(arr.length),
    };
  }

  // 2D array (matrix)
  const arr2d = value as readonly (readonly number[])[];
  const rows = arr2d.length;
  const cols = arr2d[0]?.length ?? 0;

  // Validate rectangular shape
  for (let i = 1; i < rows; i++) {
    if (arr2d[i]!.length !== cols) {
      throw new Error(
        `Inconsistent row lengths in matrix: row 0 has ${cols} cols, row ${i} has ${arr2d[i]!.length}`
      );
    }
  }

  // Flatten to column-major order for consistency with sparse matrices
  const data = new Float64Array(rows * cols);
  for (let j = 0; j < cols; j++) {
    for (let i = 0; i < rows; i++) {
      data[j * rows + i] = arr2d[i]![j]!;
    }
  }

  return {
    type: 'dense',
    data,
    shape: matrix(rows, cols),
  };
}

/**
 * Create a scalar constant.
 */
export function scalarConst(value: number): Expr {
  return constant(value);
}

/**
 * Create a vector constant.
 */
export function vectorConst(values: readonly number[] | Float64Array): Expr {
  return constant(values);
}

/**
 * Create a matrix constant.
 */
export function matrixConst(values: readonly (readonly number[])[]): Expr {
  return constant(values);
}

/**
 * Create a vector/matrix of zeros.
 *
 * @example
 * ```ts
 * const z = zeros(5);        // 5-element zero vector
 * const Z = zeros(3, 4);     // 3x4 zero matrix
 * ```
 */
export function zeros(n: number): Expr;
export function zeros(rows: number, cols: number): Expr;
export function zeros(rowsOrN: number, cols?: number): Expr {
  if (cols === undefined) {
    return {
      kind: 'constant',
      id: newExprId(),
      value: {
        type: 'dense',
        data: new Float64Array(rowsOrN),
        shape: vector(rowsOrN),
      },
    };
  }
  return {
    kind: 'constant',
    id: newExprId(),
    value: {
      type: 'dense',
      data: new Float64Array(rowsOrN * cols),
      shape: matrix(rowsOrN, cols),
    },
  };
}

/**
 * Create a vector/matrix of ones.
 *
 * @example
 * ```ts
 * const o = ones(5);        // 5-element ones vector
 * const O = ones(3, 4);     // 3x4 ones matrix
 * ```
 */
export function ones(n: number): Expr;
export function ones(rows: number, cols: number): Expr;
export function ones(rowsOrN: number, cols?: number): Expr {
  if (cols === undefined) {
    const data = new Float64Array(rowsOrN);
    data.fill(1);
    return {
      kind: 'constant',
      id: newExprId(),
      value: {
        type: 'dense',
        data,
        shape: vector(rowsOrN),
      },
    };
  }
  const data = new Float64Array(rowsOrN * cols);
  data.fill(1);
  return {
    kind: 'constant',
    id: newExprId(),
    value: {
      type: 'dense',
      data,
      shape: matrix(rowsOrN, cols),
    },
  };
}

/**
 * Create an identity matrix.
 *
 * @example
 * ```ts
 * const I = eye(3);  // 3x3 identity matrix
 * ```
 */
export function eye(n: number): Expr {
  const data = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    data[i * n + i] = 1;
  }
  return {
    kind: 'constant',
    id: newExprId(),
    value: {
      type: 'dense',
      data,
      shape: matrix(n, n),
    },
  };
}

/**
 * Check if an expression is a constant.
 */
export function isConstant(expr: Expr): expr is Extract<Expr, { kind: 'constant' }> {
  return expr.kind === 'constant';
}

/**
 * Get the numeric value of a constant expression.
 * Throws if the expression is not a constant.
 */
export function getConstantData(expr: Expr): ArrayData {
  if (expr.kind !== 'constant') {
    throw new Error(`Expected constant, got ${expr.kind}`);
  }
  return expr.value;
}

/**
 * Get the scalar value of a constant expression.
 * Throws if the expression is not a scalar constant.
 */
export function getScalarConstant(expr: Expr): number {
  if (expr.kind !== 'constant') {
    throw new Error(`Expected constant, got ${expr.kind}`);
  }
  if (expr.value.type !== 'scalar') {
    throw new Error(`Expected scalar constant, got ${expr.value.type}`);
  }
  return expr.value.value;
}
