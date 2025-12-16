import { Shape, size, scalar as scalarShape } from './shape.js';

/**
 * Unique identifier for expressions.
 * Uses a branded type for type safety.
 */
export type ExprId = number & { readonly __brand: unique symbol };

let nextExprId = 0;

/** Generate a new unique expression ID */
export function newExprId(): ExprId {
  return nextExprId++ as ExprId;
}

/** Reset the ID counter (for testing) */
export function resetExprIds(): void {
  nextExprId = 0;
}

/**
 * Array data storage.
 * Supports scalar, dense (Float64Array), and sparse (CSC) formats.
 */
export type ArrayData =
  | { readonly type: 'scalar'; readonly value: number }
  | { readonly type: 'dense'; readonly data: Float64Array; readonly shape: Shape }
  | {
      readonly type: 'sparse';
      readonly colPtr: Uint32Array;
      readonly rowIdx: Uint32Array;
      readonly values: Float64Array;
      readonly shape: Shape;
    };

/**
 * Index range for slicing expressions.
 */
export type IndexRange =
  | { type: 'single'; index: number }
  | { type: 'range'; start: number; stop: number }
  | { type: 'all' };

/**
 * Core expression data type using discriminated union.
 *
 * All expressions are immutable. The expression tree forms a DAG
 * where subexpressions can be shared.
 *
 * This is the internal data structure. Users should work with the `Expr` class
 * which wraps this type and provides a fluent API.
 */
export type ExprData =
  // === Leaf nodes ===
  | {
      readonly kind: 'variable';
      readonly id: ExprId;
      readonly shape: Shape;
      readonly name?: string;
      readonly nonneg?: boolean;
      readonly nonpos?: boolean;
    }
  | {
      readonly kind: 'constant';
      readonly id: ExprId;
      readonly value: ArrayData;
    }

  // === Affine atoms ===
  | { readonly kind: 'add'; readonly left: ExprData; readonly right: ExprData }
  | { readonly kind: 'neg'; readonly arg: ExprData }
  | { readonly kind: 'mul'; readonly left: ExprData; readonly right: ExprData } // element-wise or scalar
  | { readonly kind: 'div'; readonly left: ExprData; readonly right: ExprData } // element-wise by scalar constant
  | { readonly kind: 'matmul'; readonly left: ExprData; readonly right: ExprData }
  | { readonly kind: 'sum'; readonly arg: ExprData; readonly axis?: number }
  | { readonly kind: 'reshape'; readonly arg: ExprData; readonly shape: Shape }
  | { readonly kind: 'index'; readonly arg: ExprData; readonly ranges: readonly IndexRange[] }
  | { readonly kind: 'vstack'; readonly args: readonly ExprData[] }
  | { readonly kind: 'hstack'; readonly args: readonly ExprData[] }
  | { readonly kind: 'transpose'; readonly arg: ExprData }
  | { readonly kind: 'trace'; readonly arg: ExprData }
  | { readonly kind: 'diag'; readonly arg: ExprData }
  | { readonly kind: 'cumsum'; readonly arg: ExprData; readonly axis?: number }

  // === Nonlinear convex atoms ===
  | { readonly kind: 'norm1'; readonly arg: ExprData }
  | { readonly kind: 'norm2'; readonly arg: ExprData }
  | { readonly kind: 'normInf'; readonly arg: ExprData }
  | { readonly kind: 'abs'; readonly arg: ExprData }
  | { readonly kind: 'pos'; readonly arg: ExprData } // max(x, 0)
  | { readonly kind: 'negPart'; readonly arg: ExprData } // max(-x, 0) - negative part
  | { readonly kind: 'maximum'; readonly args: readonly ExprData[] }
  | { readonly kind: 'sumSquares'; readonly arg: ExprData }
  | { readonly kind: 'quadForm'; readonly x: ExprData; readonly P: ExprData }
  | { readonly kind: 'quadOverLin'; readonly x: ExprData; readonly y: ExprData }
  | { readonly kind: 'exp'; readonly arg: ExprData } // e^x, convex

  // === Nonlinear concave atoms ===
  | { readonly kind: 'minimum'; readonly args: readonly ExprData[] }
  | { readonly kind: 'log'; readonly arg: ExprData } // log(x), concave
  | { readonly kind: 'entropy'; readonly arg: ExprData } // -x*log(x), concave
  | { readonly kind: 'sqrt'; readonly arg: ExprData } // sqrt(x), concave
  | { readonly kind: 'power'; readonly arg: ExprData; readonly p: number }; // x^p, curvature depends on p

/**
 * Get the shape of an expression.
 */
export function exprShape(expr: ExprData): Shape {
  switch (expr.kind) {
    case 'variable':
      return expr.shape;

    case 'constant':
      return arrayDataShape(expr.value);

    case 'add':
    case 'mul':
      return exprShape(expr.left); // Assumes broadcasting already validated

    case 'neg':
    case 'abs':
    case 'pos':
    case 'negPart':
    case 'exp':
    case 'log':
    case 'entropy':
    case 'sqrt':
    case 'power':
      return exprShape(expr.arg);

    case 'div':
      return exprShape(expr.left);

    case 'matmul': {
      const leftShape = exprShape(expr.left);
      const rightShape = exprShape(expr.right);
      // (m, k) @ (k, n) -> (m, n)
      // For vectors: (k,) @ (k, n) -> (n,) or (m, k) @ (k,) -> (m,)
      if (leftShape.dims.length === 1 && rightShape.dims.length === 1) {
        return scalarShape(); // dot product
      }
      if (leftShape.dims.length === 1) {
        return { dims: [rightShape.dims[1]!] };
      }
      if (rightShape.dims.length === 1) {
        return { dims: [leftShape.dims[0]!] };
      }
      return { dims: [leftShape.dims[0]!, rightShape.dims[1]!] };
    }

    case 'sum':
      if (expr.axis === undefined) {
        return scalarShape();
      }
      // Sum along axis removes that dimension
      const argShape = exprShape(expr.arg);
      const newDims = [...argShape.dims];
      newDims.splice(expr.axis, 1);
      return { dims: newDims };

    case 'reshape':
      return expr.shape;

    case 'index': {
      const baseShape = exprShape(expr.arg);
      const newDims: number[] = [];

      for (let i = 0; i < expr.ranges.length; i++) {
        const range = expr.ranges[i]!;
        const dim = baseShape.dims[i] ?? 1;

        if (range.type === 'single') {
          // Single index removes dimension (unless we want to keep it)
          continue;
        } else if (range.type === 'range') {
          newDims.push(range.stop - range.start);
        } else {
          newDims.push(dim);
        }
      }

      if (newDims.length === 0) return scalarShape();
      return { dims: newDims };
    }

    case 'vstack': {
      const shapes = expr.args.map(exprShape);
      const totalRows = shapes.reduce((sum, s) => sum + (s.dims[0] ?? 1), 0);
      const cols = shapes[0]?.dims[1] ?? 1;
      return { dims: [totalRows, cols] };
    }

    case 'hstack': {
      const shapes = expr.args.map(exprShape);
      const rows = shapes[0]?.dims[0] ?? 1;
      const totalCols = shapes.reduce((sum, s) => sum + (s.dims[1] ?? 1), 0);
      return { dims: [rows, totalCols] };
    }

    case 'transpose': {
      const argShape = exprShape(expr.arg);
      if (argShape.dims.length <= 1) return argShape;
      return { dims: [argShape.dims[1]!, argShape.dims[0]!] };
    }

    case 'trace':
    case 'norm1':
    case 'norm2':
    case 'normInf':
    case 'sumSquares':
    case 'quadForm':
    case 'quadOverLin':
      return scalarShape();

    case 'diag': {
      const argShape = exprShape(expr.arg);
      if (argShape.dims.length === 1) {
        // Vector -> diagonal matrix
        const n = argShape.dims[0]!;
        return { dims: [n, n] };
      }
      // Matrix -> diagonal vector
      const n = Math.min(argShape.dims[0]!, argShape.dims[1]!);
      return { dims: [n] };
    }

    case 'cumsum':
      // Cumulative sum preserves shape
      return exprShape(expr.arg);

    case 'maximum':
    case 'minimum':
      // Element-wise maximum/minimum preserves shape
      return exprShape(expr.args[0]!);
  }
}

/**
 * Get all variable IDs referenced in an expression.
 */
export function exprVariables(expr: ExprData): Set<ExprId> {
  const vars = new Set<ExprId>();
  collectVariables(expr, vars);
  return vars;
}

function collectVariables(expr: ExprData, vars: Set<ExprId>): void {
  switch (expr.kind) {
    case 'variable':
      vars.add(expr.id);
      break;

    case 'constant':
      break;

    case 'add':
    case 'mul':
    case 'div':
    case 'matmul':
      collectVariables(expr.left, vars);
      collectVariables(expr.right, vars);
      break;

    case 'quadForm':
      collectVariables(expr.x, vars);
      collectVariables(expr.P, vars);
      break;

    case 'quadOverLin':
      collectVariables(expr.x, vars);
      collectVariables(expr.y, vars);
      break;

    case 'neg':
    case 'sum':
    case 'reshape':
    case 'index':
    case 'transpose':
    case 'trace':
    case 'diag':
    case 'cumsum':
    case 'norm1':
    case 'norm2':
    case 'normInf':
    case 'abs':
    case 'pos':
    case 'negPart':
    case 'sumSquares':
    case 'exp':
    case 'log':
    case 'entropy':
    case 'sqrt':
    case 'power':
      collectVariables(expr.arg, vars);
      break;

    case 'vstack':
    case 'hstack':
    case 'maximum':
    case 'minimum':
      for (const arg of expr.args) {
        collectVariables(arg, vars);
      }
      break;
  }
}

/**
 * Check if an expression is constant (contains no variables).
 */
export function isConstantExpr(expr: ExprData): boolean {
  return exprVariables(expr).size === 0;
}

/** Get the shape of ArrayData */
function arrayDataShape(data: ArrayData): Shape {
  switch (data.type) {
    case 'scalar':
      return scalarShape();
    case 'dense':
    case 'sparse':
      return data.shape;
  }
}

/** Get the total size of ArrayData */
export function arrayDataSize(data: ArrayData): number {
  return size(arrayDataShape(data));
}

/** Check if ArrayData represents a scalar */
export function isScalarData(data: ArrayData): boolean {
  return data.type === 'scalar' || size(arrayDataShape(data)) === 1;
}

/** Get scalar value from ArrayData (throws if not scalar) */
export function getScalarValue(data: ArrayData): number {
  if (data.type === 'scalar') return data.value;
  if (size(arrayDataShape(data)) === 1) {
    if (data.type === 'dense') return data.data[0]!;
    return data.values[0]!;
  }
  throw new Error('ArrayData is not a scalar');
}
