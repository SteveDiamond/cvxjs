/**
 * Compressed Sparse Column (CSC) matrix format.
 *
 * CSC stores sparse matrices efficiently by storing:
 * - colPtr: Index into rowIdx/values for start of each column
 * - rowIdx: Row indices of non-zero elements
 * - values: Non-zero values
 *
 * For an m×n matrix with nnz non-zeros:
 * - colPtr has length n + 1
 * - rowIdx and values have length nnz
 */
export interface CscMatrix {
  /** Number of rows */
  readonly nrows: number;
  /** Number of columns */
  readonly ncols: number;
  /** Column pointers (length ncols + 1) */
  readonly colPtr: Uint32Array;
  /** Row indices (length nnz) */
  readonly rowIdx: Uint32Array;
  /** Non-zero values (length nnz) */
  readonly values: Float64Array;
}

/**
 * Create an empty CSC matrix.
 */
export function cscEmpty(nrows: number, ncols: number): CscMatrix {
  const colPtr = new Uint32Array(ncols + 1);
  return {
    nrows,
    ncols,
    colPtr,
    rowIdx: new Uint32Array(0),
    values: new Float64Array(0),
  };
}

/**
 * Create a CSC identity matrix.
 */
export function cscIdentity(n: number): CscMatrix {
  const colPtr = new Uint32Array(n + 1);
  const rowIdx = new Uint32Array(n);
  const values = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    colPtr[i] = i;
    rowIdx[i] = i;
    values[i] = 1;
  }
  colPtr[n] = n;

  return { nrows: n, ncols: n, colPtr, rowIdx, values };
}

/**
 * Create a CSC matrix of zeros with specified shape.
 */
export function cscZeros(nrows: number, ncols: number): CscMatrix {
  return cscEmpty(nrows, ncols);
}

/**
 * Create CSC matrix from triplets (row, col, value).
 *
 * Duplicates at the same position are summed.
 */
export function cscFromTriplets(
  nrows: number,
  ncols: number,
  rows: number[],
  cols: number[],
  vals: number[]
): CscMatrix {
  if (rows.length !== cols.length || cols.length !== vals.length) {
    throw new Error('Triplet arrays must have same length');
  }

  if (rows.length === 0) {
    return cscEmpty(nrows, ncols);
  }

  // Create triplets and sort by column, then row
  const triplets: Array<{ r: number; c: number; v: number }> = [];
  for (let i = 0; i < rows.length; i++) {
    if (vals[i] !== 0) {
      triplets.push({ r: rows[i]!, c: cols[i]!, v: vals[i]! });
    }
  }

  // Sort by column, then by row
  triplets.sort((a, b) => {
    if (a.c !== b.c) return a.c - b.c;
    return a.r - b.r;
  });

  // Merge duplicates and build CSC
  const mergedRows: number[] = [];
  const mergedVals: number[] = [];
  const colCounts = new Array(ncols).fill(0);

  for (let i = 0; i < triplets.length; i++) {
    const t = triplets[i]!;

    // Check if same position as previous
    if (i > 0 && triplets[i - 1]!.r === t.r && triplets[i - 1]!.c === t.c) {
      // Add to previous value
      mergedVals[mergedVals.length - 1]! += t.v;
    } else {
      mergedRows.push(t.r);
      mergedVals.push(t.v);
      colCounts[t.c]!++;
    }
  }

  // Build colPtr
  const colPtr = new Uint32Array(ncols + 1);
  colPtr[0] = 0;
  for (let c = 0; c < ncols; c++) {
    colPtr[c + 1] = colPtr[c]! + colCounts[c]!;
  }

  // Build rowIdx and values in column order
  const nnz = mergedRows.length;
  const rowIdx = new Uint32Array(nnz);
  const values = new Float64Array(nnz);

  // Re-sort merged data by column
  const sortedIdx: number[] = [];
  for (let i = 0; i < triplets.length; i++) {
    // Skip duplicates we already merged
    if (i > 0 && triplets[i - 1]!.r === triplets[i]!.r && triplets[i - 1]!.c === triplets[i]!.c) {
      continue;
    }
    sortedIdx.push(i);
  }

  // Use the merged values directly
  for (let i = 0; i < nnz; i++) {
    rowIdx[i] = mergedRows[i]!;
    values[i] = mergedVals[i]!;
  }

  return { nrows, ncols, colPtr, rowIdx, values };
}

/**
 * Create CSC matrix from dense array (column-major order).
 */
export function cscFromDense(nrows: number, ncols: number, data: Float64Array): CscMatrix {
  const rows: number[] = [];
  const cols: number[] = [];
  const vals: number[] = [];

  for (let c = 0; c < ncols; c++) {
    for (let r = 0; r < nrows; r++) {
      const v = data[c * nrows + r];
      if (v !== undefined && v !== 0) {
        rows.push(r);
        cols.push(c);
        vals.push(v);
      }
    }
  }

  return cscFromTriplets(nrows, ncols, rows, cols, vals);
}

/**
 * Number of non-zero elements.
 */
export function cscNnz(A: CscMatrix): number {
  return A.values.length;
}

/**
 * Get element at (row, col). Returns 0 for elements not stored.
 */
export function cscGet(A: CscMatrix, row: number, col: number): number {
  const start = A.colPtr[col]!;
  const end = A.colPtr[col + 1]!;

  for (let i = start; i < end; i++) {
    if (A.rowIdx[i] === row) {
      return A.values[i]!;
    }
  }
  return 0;
}

/**
 * Scale matrix by scalar: result = scalar * A
 */
export function cscScale(A: CscMatrix, scalar: number): CscMatrix {
  if (scalar === 0) {
    return cscZeros(A.nrows, A.ncols);
  }

  const values = new Float64Array(A.values.length);
  for (let i = 0; i < A.values.length; i++) {
    values[i] = scalar * A.values[i]!;
  }

  return {
    nrows: A.nrows,
    ncols: A.ncols,
    colPtr: A.colPtr, // Shared (immutable)
    rowIdx: A.rowIdx, // Shared (immutable)
    values,
  };
}

/**
 * Add two matrices: result = A + B
 */
export function cscAdd(A: CscMatrix, B: CscMatrix): CscMatrix {
  if (A.nrows !== B.nrows || A.ncols !== B.ncols) {
    throw new Error(
      `Cannot add matrices with different shapes: ${A.nrows}×${A.ncols} vs ${B.nrows}×${B.ncols}`
    );
  }

  const rows: number[] = [];
  const cols: number[] = [];
  const vals: number[] = [];

  for (let c = 0; c < A.ncols; c++) {
    // Get elements from A
    for (let i = A.colPtr[c]!; i < A.colPtr[c + 1]!; i++) {
      rows.push(A.rowIdx[i]!);
      cols.push(c);
      vals.push(A.values[i]!);
    }

    // Get elements from B
    for (let i = B.colPtr[c]!; i < B.colPtr[c + 1]!; i++) {
      rows.push(B.rowIdx[i]!);
      cols.push(c);
      vals.push(B.values[i]!);
    }
  }

  return cscFromTriplets(A.nrows, A.ncols, rows, cols, vals);
}

/**
 * Subtract two matrices: result = A - B
 */
export function cscSub(A: CscMatrix, B: CscMatrix): CscMatrix {
  return cscAdd(A, cscScale(B, -1));
}

/**
 * Vertically stack matrices: result = [A; B]
 */
export function cscVstack(A: CscMatrix, B: CscMatrix): CscMatrix {
  if (A.ncols !== B.ncols) {
    throw new Error(
      `Cannot vstack matrices with different column counts: ${A.ncols} vs ${B.ncols}`
    );
  }

  const nrows = A.nrows + B.nrows;
  const ncols = A.ncols;
  const rows: number[] = [];
  const cols: number[] = [];
  const vals: number[] = [];

  // Add elements from A
  for (let c = 0; c < ncols; c++) {
    for (let i = A.colPtr[c]!; i < A.colPtr[c + 1]!; i++) {
      rows.push(A.rowIdx[i]!);
      cols.push(c);
      vals.push(A.values[i]!);
    }
  }

  // Add elements from B (shifted down by A.nrows)
  for (let c = 0; c < ncols; c++) {
    for (let i = B.colPtr[c]!; i < B.colPtr[c + 1]!; i++) {
      rows.push(B.rowIdx[i]! + A.nrows);
      cols.push(c);
      vals.push(B.values[i]!);
    }
  }

  return cscFromTriplets(nrows, ncols, rows, cols, vals);
}

/**
 * Horizontally stack matrices: result = [A, B]
 */
export function cscHstack(A: CscMatrix, B: CscMatrix): CscMatrix {
  if (A.nrows !== B.nrows) {
    throw new Error(`Cannot hstack matrices with different row counts: ${A.nrows} vs ${B.nrows}`);
  }

  const nrows = A.nrows;
  const ncols = A.ncols + B.ncols;

  // Build new colPtr
  const colPtr = new Uint32Array(ncols + 1);
  colPtr[0] = 0;

  // Copy A's column structure
  for (let c = 0; c <= A.ncols; c++) {
    colPtr[c] = A.colPtr[c]!;
  }

  // Copy B's column structure (shifted)
  const aOffset = A.values.length;
  for (let c = 0; c <= B.ncols; c++) {
    colPtr[A.ncols + c] = aOffset + B.colPtr[c]!;
  }

  // Concatenate rowIdx and values
  const nnz = A.values.length + B.values.length;
  const rowIdx = new Uint32Array(nnz);
  const values = new Float64Array(nnz);

  rowIdx.set(A.rowIdx, 0);
  rowIdx.set(B.rowIdx, A.values.length);
  values.set(A.values, 0);
  values.set(B.values, A.values.length);

  return { nrows, ncols, colPtr, rowIdx, values };
}

/**
 * Transpose a matrix.
 */
export function cscTranspose(A: CscMatrix): CscMatrix {
  const rows: number[] = [];
  const cols: number[] = [];
  const vals: number[] = [];

  for (let c = 0; c < A.ncols; c++) {
    for (let i = A.colPtr[c]!; i < A.colPtr[c + 1]!; i++) {
      // Swap row and column
      rows.push(c);
      cols.push(A.rowIdx[i]!);
      vals.push(A.values[i]!);
    }
  }

  return cscFromTriplets(A.ncols, A.nrows, rows, cols, vals);
}

/**
 * Matrix-vector multiplication: result = A * x
 */
export function cscMulVec(A: CscMatrix, x: Float64Array): Float64Array {
  if (A.ncols !== x.length) {
    throw new Error(`Cannot multiply: matrix has ${A.ncols} cols, vector has ${x.length} elements`);
  }

  const result = new Float64Array(A.nrows);

  for (let c = 0; c < A.ncols; c++) {
    const xc = x[c]!;
    for (let i = A.colPtr[c]!; i < A.colPtr[c + 1]!; i++) {
      const row = A.rowIdx[i]!;
      result[row] = (result[row] ?? 0) + A.values[i]! * xc;
    }
  }

  return result;
}

/**
 * Matrix-matrix multiplication: result = A * B
 */
export function cscMulMat(A: CscMatrix, B: CscMatrix): CscMatrix {
  if (A.ncols !== B.nrows) {
    throw new Error(`Cannot multiply: A has ${A.ncols} cols, B has ${B.nrows} rows`);
  }

  const rows: number[] = [];
  const cols: number[] = [];
  const vals: number[] = [];

  // For each column of B
  for (let c = 0; c < B.ncols; c++) {
    // Accumulator for result column
    const colResult = new Float64Array(A.nrows);

    // For each element in B's column
    for (let i = B.colPtr[c]!; i < B.colPtr[c + 1]!; i++) {
      const bRow = B.rowIdx[i]!;
      const bVal = B.values[i]!;

      // Add A's column bRow scaled by bVal
      for (let j = A.colPtr[bRow]!; j < A.colPtr[bRow + 1]!; j++) {
        const aRow = A.rowIdx[j]!;
        colResult[aRow] = (colResult[aRow] ?? 0) + A.values[j]! * bVal;
      }
    }

    // Store non-zeros in result column
    for (let r = 0; r < A.nrows; r++) {
      if (colResult[r] !== 0) {
        rows.push(r);
        cols.push(c);
        vals.push(colResult[r]!);
      }
    }
  }

  return cscFromTriplets(A.nrows, B.ncols, rows, cols, vals);
}

/**
 * Multiply A' * B (transpose of A times B).
 * More efficient than transposing first: O(nnz(A) * nnz(B))
 */
export function cscMulMatTransposeLeft(A: CscMatrix, B: CscMatrix): CscMatrix {
  if (A.nrows !== B.nrows) {
    throw new Error(`Cannot multiply A' * B: A has ${A.nrows} rows, B has ${B.nrows} rows`);
  }

  const rows: number[] = [];
  const cols: number[] = [];
  const vals: number[] = [];

  // A' has dimensions (ncols x nrows), result has dimensions (A.ncols x B.ncols)
  // (A')[i, :] = A[:, i] (the i-th column of A becomes i-th row of A')

  // For each column of B
  for (let c = 0; c < B.ncols; c++) {
    // Accumulator for result column (length = A.ncols)
    const colResult = new Float64Array(A.ncols);

    // For each non-zero in B's column c
    for (let i = B.colPtr[c]!; i < B.colPtr[c + 1]!; i++) {
      const bRow = B.rowIdx[i]!; // row in B = row in A (for A' * B)
      const bVal = B.values[i]!;

      // Find elements in A that share this row index
      // Need to search all columns of A for row bRow
      // For each column j of A (which becomes row j of A')
      for (let j = 0; j < A.ncols; j++) {
        // Look for row bRow in column j of A
        for (let k = A.colPtr[j]!; k < A.colPtr[j + 1]!; k++) {
          if (A.rowIdx[k] === bRow) {
            // A[bRow, j] exists, contributes A[bRow, j] * B[bRow, c] to result[j, c]
            colResult[j] = (colResult[j] ?? 0) + A.values[k]! * bVal;
          }
        }
      }
    }

    // Store non-zeros in result column
    for (let r = 0; r < A.ncols; r++) {
      if (colResult[r] !== 0) {
        rows.push(r);
        cols.push(c);
        vals.push(colResult[r]!);
      }
    }
  }

  return cscFromTriplets(A.ncols, B.ncols, rows, cols, vals);
}

/**
 * Create a diagonal matrix from vector.
 */
export function cscDiag(v: Float64Array): CscMatrix {
  const n = v.length;
  const nnz = v.filter((x) => x !== 0).length;

  const colPtr = new Uint32Array(n + 1);
  const rowIdx = new Uint32Array(nnz);
  const values = new Float64Array(nnz);

  let idx = 0;
  for (let i = 0; i < n; i++) {
    colPtr[i] = idx;
    if (v[i] !== 0) {
      rowIdx[idx] = i;
      values[idx] = v[i]!;
      idx++;
    }
  }
  colPtr[n] = idx;

  return { nrows: n, ncols: n, colPtr, rowIdx, values };
}

/**
 * Convert CSC matrix to dense array (column-major order).
 */
export function cscToDense(A: CscMatrix): Float64Array {
  const result = new Float64Array(A.nrows * A.ncols);

  for (let c = 0; c < A.ncols; c++) {
    for (let i = A.colPtr[c]!; i < A.colPtr[c + 1]!; i++) {
      result[c * A.nrows + A.rowIdx[i]!] = A.values[i]!;
    }
  }

  return result;
}

/**
 * Clone a CSC matrix.
 */
export function cscClone(A: CscMatrix): CscMatrix {
  return {
    nrows: A.nrows,
    ncols: A.ncols,
    colPtr: new Uint32Array(A.colPtr),
    rowIdx: new Uint32Array(A.rowIdx),
    values: new Float64Array(A.values),
  };
}

/**
 * Check if two matrices are equal.
 */
export function cscEquals(A: CscMatrix, B: CscMatrix, tol = 1e-10): boolean {
  if (A.nrows !== B.nrows || A.ncols !== B.ncols) {
    return false;
  }

  if (A.values.length !== B.values.length) {
    return false;
  }

  for (let c = 0; c < A.ncols; c++) {
    if (A.colPtr[c] !== B.colPtr[c]) {
      return false;
    }
  }

  if (A.colPtr[A.ncols] !== B.colPtr[B.ncols]) {
    return false;
  }

  for (let i = 0; i < A.values.length; i++) {
    if (A.rowIdx[i] !== B.rowIdx[i]) {
      return false;
    }
    if (Math.abs(A.values[i]! - B.values[i]!) > tol) {
      return false;
    }
  }

  return true;
}
