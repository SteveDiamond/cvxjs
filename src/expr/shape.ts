/**
 * Shape representation for expressions.
 *
 * Shapes are immutable and describe the dimensions of arrays.
 * - Scalar: dims = []
 * - Vector: dims = [n]
 * - Matrix: dims = [rows, cols]
 */
export interface Shape {
  readonly dims: readonly number[];
}

/** Create a scalar shape */
export function scalar(): Shape {
  return { dims: [] };
}

/** Create a vector shape */
export function vector(n: number): Shape {
  if (n < 1) throw new Error(`Vector size must be >= 1, got ${n}`);
  return { dims: [n] };
}

/** Create a matrix shape */
export function matrix(rows: number, cols: number): Shape {
  if (rows < 1 || cols < 1) {
    throw new Error(`Matrix dimensions must be >= 1, got ${rows}x${cols}`);
  }
  return { dims: [rows, cols] };
}

/** Total number of elements in the shape */
export function size(shape: Shape): number {
  return shape.dims.reduce((a, b) => a * b, 1);
}

/** Number of rows (1 for scalars and vectors) */
export function rows(shape: Shape): number {
  if (shape.dims.length === 0) return 1;
  if (shape.dims.length === 1) return shape.dims[0]!;
  return shape.dims[0]!;
}

/** Number of columns (1 for scalars and vectors) */
export function cols(shape: Shape): number {
  if (shape.dims.length <= 1) return 1;
  return shape.dims[1]!;
}

/** Check if shape is scalar */
export function isScalar(shape: Shape): boolean {
  return shape.dims.length === 0;
}

/** Check if shape is a vector (1D) */
export function isVector(shape: Shape): boolean {
  return shape.dims.length === 1;
}

/** Check if shape is a matrix (2D) */
export function isMatrix(shape: Shape): boolean {
  return shape.dims.length === 2;
}

/** Check if two shapes are equal */
export function shapeEquals(a: Shape, b: Shape): boolean {
  if (a.dims.length !== b.dims.length) return false;
  for (let i = 0; i < a.dims.length; i++) {
    if (a.dims[i] !== b.dims[i]) return false;
  }
  return true;
}

/** Format shape as string for debugging */
export function shapeToString(shape: Shape): string {
  if (shape.dims.length === 0) return 'scalar';
  return `(${shape.dims.join(', ')})`;
}

/**
 * Compute broadcast shape for two shapes.
 * Returns null if shapes are not broadcastable.
 */
export function broadcastShape(a: Shape, b: Shape): Shape | null {
  // Scalars broadcast to anything
  if (a.dims.length === 0) return b;
  if (b.dims.length === 0) return a;

  // Must have same number of dimensions for now
  if (a.dims.length !== b.dims.length) return null;

  const result: number[] = [];
  for (let i = 0; i < a.dims.length; i++) {
    const da = a.dims[i]!;
    const db = b.dims[i]!;

    if (da === db) {
      result.push(da);
    } else if (da === 1) {
      result.push(db);
    } else if (db === 1) {
      result.push(da);
    } else {
      return null; // Not broadcastable
    }
  }

  return { dims: result };
}

/**
 * Normalize shape input to a Shape object.
 * Accepts: number (vector), [n] (vector), [m, n] (matrix), or Shape
 */
export function normalizeShape(
  input: number | readonly [number] | readonly [number, number] | Shape
): Shape {
  if (typeof input === 'number') {
    return vector(input);
  }
  if ('dims' in input) {
    return input;
  }
  if (input.length === 1) {
    return vector(input[0]);
  }
  return matrix(input[0], input[1]);
}
