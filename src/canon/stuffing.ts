import { ExprId } from '../expr/index.js';
import { LinExpr } from './lin-expr.js';
import { QuadExpr, parseQuadCoeffKey } from './quad-expr.js';
import { ConeConstraint, ConeDims, emptyConeDims } from './cone-constraint.js';
import { AuxVar } from './canonicalizer.js';
import { CscMatrix, cscZeros, cscFromTriplets, cscVstack, cscAdd } from '../sparse/index.js';

/**
 * Variable mapping from expression IDs to solver variable indices.
 */
export interface VariableMap {
  /** Map from variable ID to (start column, size) */
  readonly idToCol: ReadonlyMap<ExprId, { start: number; size: number }>;
  /** Total number of optimization variables */
  readonly totalVars: number;
}

/**
 * Stuffed problem in standard solver form.
 *
 * minimize    (1/2) x' P x + q' x
 * subject to  A x + s = b
 *             s in K
 *
 * where K is the cone product specified by coneDims.
 */
export interface StuffedProblem {
  /** Quadratic cost matrix P (n x n, upper triangular) */
  readonly P: CscMatrix;
  /** Linear cost vector q (length n) */
  readonly q: Float64Array;
  /** Constraint matrix A (m x n) */
  readonly A: CscMatrix;
  /** Constraint vector b (length m) */
  readonly b: Float64Array;
  /** Cone dimensions specifying K */
  readonly coneDims: ConeDims;
  /** Variable mapping */
  readonly varMap: VariableMap;
  /** Number of variables */
  readonly nVars: number;
  /** Number of constraints */
  readonly nConstraints: number;
  /** Constant offset in objective */
  readonly objectiveOffset: number;
}

/**
 * Build variable mapping from original and auxiliary variables.
 */
export function buildVariableMap(
  originalVarIds: Set<ExprId>,
  originalVarSizes: Map<ExprId, number>,
  auxVars: AuxVar[]
): VariableMap {
  const idToCol = new Map<ExprId, { start: number; size: number }>();
  let col = 0;

  // Map original variables
  for (const varId of originalVarIds) {
    const size = originalVarSizes.get(varId) ?? 1;
    idToCol.set(varId, { start: col, size });
    col += size;
  }

  // Map auxiliary variables
  for (const aux of auxVars) {
    idToCol.set(aux.id, { start: col, size: aux.size });
    col += aux.size;
  }

  return { idToCol, totalVars: col };
}

/**
 * Stuff a linear expression into coefficient row(s) of A matrix.
 *
 * @param linExpr - Linear expression to stuff
 * @param varMap - Variable mapping
 * @param negate - Whether to negate (for converting >= to <=)
 * @returns Coefficient matrix row(s) and constant vector
 */
export function stuffLinExpr(
  linExpr: LinExpr,
  varMap: VariableMap,
  negate = false
): { A: CscMatrix; b: Float64Array } {
  const nrows = linExpr.rows;
  const ncols = varMap.totalVars;
  const sign = negate ? -1 : 1;

  // Collect triplets for A
  const aRows: number[] = [];
  const aCols: number[] = [];
  const aVals: number[] = [];

  for (const [varId, coeff] of linExpr.coeffs) {
    const mapping = varMap.idToCol.get(varId);
    if (!mapping) {
      throw new Error(`Variable ${varId} not found in variable map`);
    }

    // Copy coefficients with column offset
    for (let c = 0; c < coeff.ncols; c++) {
      for (let i = coeff.colPtr[c]!; i < coeff.colPtr[c + 1]!; i++) {
        aRows.push(coeff.rowIdx[i]!);
        aCols.push(mapping.start + c);
        aVals.push(sign * coeff.values[i]!);
      }
    }
  }

  const A = cscFromTriplets(nrows, ncols, aRows, aCols, aVals);

  // Compute b = -constant (for Ax + b = 0 form, we want Ax = -constant)
  const b = new Float64Array(nrows);
  for (let i = 0; i < nrows; i++) {
    b[i] = -sign * linExpr.constant[i]!;
  }

  return { A, b };
}

/**
 * Stuff objective linear expression into q vector.
 */
export function stuffObjective(linExpr: LinExpr, varMap: VariableMap): Float64Array {
  if (linExpr.rows !== 1) {
    throw new Error(`Objective must be scalar, got ${linExpr.rows} rows`);
  }

  const q = new Float64Array(varMap.totalVars);

  for (const [varId, coeff] of linExpr.coeffs) {
    const mapping = varMap.idToCol.get(varId);
    if (!mapping) {
      throw new Error(`Variable ${varId} not found in variable map`);
    }

    // coeff is 1 x varSize, extract values
    for (let c = 0; c < coeff.ncols; c++) {
      for (let i = coeff.colPtr[c]!; i < coeff.colPtr[c + 1]!; i++) {
        q[mapping.start + c] = coeff.values[i]!;
      }
    }
  }

  return q;
}

/**
 * Stuff quadratic objective into P matrix.
 *
 * The P matrix represents the quadratic part of the objective.
 * Clarabel expects the form (1/2) x' P x, so if we have x' Q x,
 * we need P = 2 * Q.
 */
export function stuffQuadraticObjective(
  quadExpr: QuadExpr,
  varMap: VariableMap
): CscMatrix {
  const nVars = varMap.totalVars;

  // Start with zero matrix
  let P = cscZeros(nVars, nVars);

  for (const [key, coeff] of quadExpr.quadCoeffs) {
    const [varId1, varId2] = parseQuadCoeffKey(key);
    const mapping1 = varMap.idToCol.get(varId1);
    const mapping2 = varMap.idToCol.get(varId2);

    if (!mapping1 || !mapping2) {
      throw new Error(`Variable not found in variable map: ${varId1} or ${varId2}`);
    }

    // coeff is the P_ij block for variables i and j
    // For single variable case (varId1 === varId2), coeff is the full P matrix
    // Clarabel expects (1/2) x' P x, so we scale by 2

    // Build triplets for this block
    const rows: number[] = [];
    const cols: number[] = [];
    const vals: number[] = [];

    for (let c = 0; c < coeff.ncols; c++) {
      for (let i = coeff.colPtr[c]!; i < coeff.colPtr[c + 1]!; i++) {
        const r = coeff.rowIdx[i]!;
        const v = coeff.values[i]!;

        // Map to global variable indices
        const globalRow = mapping1.start + r;
        const globalCol = mapping2.start + c;

        // Scale by 2 for Clarabel's (1/2) x' P x form
        rows.push(globalRow);
        cols.push(globalCol);
        vals.push(2 * v);
      }
    }

    const block = cscFromTriplets(nVars, nVars, rows, cols, vals);
    P = cscAdd(P, block);
  }

  return P;
}

/**
 * Stuff problem into standard solver form.
 */
export function stuffProblem(
  objectiveLinExpr: LinExpr,
  coneConstraints: ConeConstraint[],
  varMap: VariableMap,
  objectiveOffset: number,
  objectiveQuadExpr?: QuadExpr
): StuffedProblem {
  const nVars = varMap.totalVars;

  // Stuff objective
  const q = stuffObjective(objectiveLinExpr, varMap);

  // Build P matrix from quadratic expression if present
  const P = objectiveQuadExpr
    ? stuffQuadraticObjective(objectiveQuadExpr, varMap)
    : cscZeros(nVars, nVars);

  // Organize constraints by cone type
  const zeroConstraints: ConeConstraint[] = [];
  const nonnegConstraints: ConeConstraint[] = [];
  const socConstraints: ConeConstraint[] = [];
  const expConstraints: ConeConstraint[] = [];
  const powerConstraints: ConeConstraint[] = [];

  for (const c of coneConstraints) {
    switch (c.kind) {
      case 'zero':
        zeroConstraints.push(c);
        break;
      case 'nonneg':
        nonnegConstraints.push(c);
        break;
      case 'soc':
        socConstraints.push(c);
        break;
      case 'exp':
        expConstraints.push(c);
        break;
      case 'power':
        powerConstraints.push(c);
        break;
    }
  }

  // Build A and b by stacking constraint rows
  const As: CscMatrix[] = [];
  const bs: Float64Array[] = [];
  const coneDims = emptyConeDims();

  // Zero cone (equality constraints): Ax + b = 0
  for (const c of zeroConstraints) {
    if (c.kind !== 'zero') continue;
    const { A, b } = stuffLinExpr(c.a, varMap, false);
    As.push(A);
    bs.push(b);
    coneDims.zero += c.a.rows;
  }

  // Nonnegative cone (inequality constraints): expr >= 0
  // Clarabel standard form: Ax + s = b, s >= 0, means s = b - Ax >= 0, i.e., Ax <= b
  // We have: a_coeffs * x + a_const >= 0
  // Rewrite: -a_coeffs * x <= a_const
  // So: A_clarabel = -a_coeffs, b_clarabel = a_const
  // stuffLinExpr with negate=true gives: A = -a_coeffs, b = -(-a_const) = a_const ✗
  // Actually stuffLinExpr computes b = -constant, so:
  //   negate=true: A = -a_coeffs, b = -(-1) * a_const = a_const ✓
  for (const c of nonnegConstraints) {
    if (c.kind !== 'nonneg') continue;
    const { A, b } = stuffLinExpr(c.a, varMap, true); // NEGATE for correct sign
    As.push(A);
    bs.push(b);
    coneDims.nonneg += c.a.rows;
  }

  // SOC constraints: ||x|| <= t
  // Clarabel form: Ax + s = b, s in SOC means s[0] >= ||s[1:]||
  // We want slack s = [t; x] so that t >= ||x||
  // From Ax + s = b: s = b - Ax
  // For s = [t; x], we need b - Ax = [t; x]
  // If t and x are variables with LinExpr A_t*vars + c_t and A_x*vars + c_x:
  //   s = b - A_clarabel*vars
  //   For s = [t; x] = [A_t*vars + c_t; A_x*vars + c_x]
  //   We need A_clarabel = -[A_t; A_x], b = [c_t; c_x]
  // stuffLinExpr with negate=true gives: A = -coeffs, b = constant
  for (const c of socConstraints) {
    if (c.kind !== 'soc') continue;
    const { A: At, b: bt } = stuffLinExpr(c.t, varMap, true); // NEGATE for correct slack
    const { A: Ax, b: bx } = stuffLinExpr(c.x, varMap, true); // NEGATE for correct slack

    // Stack [t; x]
    As.push(cscVstack(At, Ax));
    const b = new Float64Array(c.t.rows + c.x.rows);
    b.set(bt, 0);
    b.set(bx, c.t.rows);
    bs.push(b);

    coneDims.soc.push(c.t.rows + c.x.rows);
  }

  // Exponential cone constraints: s = [x; y; z] in ExpCone
  // From Ax + s = b: s = b - A_clarabel*vars
  // For s = [x; y; z], we need negate=true like SOC
  for (const c of expConstraints) {
    if (c.kind !== 'exp') continue;
    const { A: Ax, b: bx } = stuffLinExpr(c.x, varMap, true); // NEGATE for correct slack
    const { A: Ay, b: by } = stuffLinExpr(c.y, varMap, true); // NEGATE for correct slack
    const { A: Az, b: bz } = stuffLinExpr(c.z, varMap, true); // NEGATE for correct slack

    // Stack [x; y; z]
    As.push(cscVstack(cscVstack(Ax, Ay), Az));
    const b = new Float64Array(3);
    b[0] = bx[0]!;
    b[1] = by[0]!;
    b[2] = bz[0]!;
    bs.push(b);

    coneDims.exp += 1;
  }

  // Power cone constraints: s = [x; y; z] in PowerCone(alpha)
  // From Ax + s = b: s = b - A_clarabel*vars
  // For s = [x; y; z], we need negate=true like SOC
  for (const c of powerConstraints) {
    if (c.kind !== 'power') continue;
    const { A: Ax, b: bx } = stuffLinExpr(c.x, varMap, true); // NEGATE for correct slack
    const { A: Ay, b: by } = stuffLinExpr(c.y, varMap, true); // NEGATE for correct slack
    const { A: Az, b: bz } = stuffLinExpr(c.z, varMap, true); // NEGATE for correct slack

    // Stack [x; y; z]
    As.push(cscVstack(cscVstack(Ax, Ay), Az));
    const b = new Float64Array(3);
    b[0] = bx[0]!;
    b[1] = by[0]!;
    b[2] = bz[0]!;
    bs.push(b);

    coneDims.power.push(c.alpha);
  }

  // Combine all constraint matrices
  let A: CscMatrix;
  let b: Float64Array;

  if (As.length === 0) {
    A = cscZeros(0, nVars);
    b = new Float64Array(0);
  } else {
    A = As[0]!;
    b = bs[0]!;
    for (let i = 1; i < As.length; i++) {
      A = cscVstack(A, As[i]!);
      const newB = new Float64Array(b.length + bs[i]!.length);
      newB.set(b, 0);
      newB.set(bs[i]!, b.length);
      b = newB;
    }
  }

  return {
    P,
    q,
    A,
    b,
    coneDims,
    varMap,
    nVars,
    nConstraints: A.nrows,
    objectiveOffset,
  };
}
