/**
 * cvxjs - Disciplined Convex Programming in TypeScript
 *
 * @example
 * ```ts
 * import { variable, sum, norm2, Problem } from 'cvxjs';
 *
 * // Create a variable
 * const x = variable(5);
 *
 * // Build and solve a problem
 * const solution = await Problem.minimize(sum(x))
 *   .subjectTo([x.ge(1)])
 *   .solve();
 *
 * console.log('Optimal value:', solution.value);
 * ```
 *
 * @packageDocumentation
 */

// === Expression Types ===
export type { Shape, ExprId, ArrayData, IndexRange, Expr } from './expr/index.js';

// === Shape Utilities ===
export {
  scalar,
  vector,
  matrix,
  size,
  rows,
  cols,
  isScalar,
  isVector,
  isMatrix,
  shapeEquals,
  shapeToString,
  normalizeShape,
} from './expr/index.js';

// === Variable Creation ===
export {
  variable,
  scalarVar,
  vectorVar,
  matrixVar,
  VariableBuilder,
  isVariable,
} from './expr/index.js';

// === Constant Creation ===
export { constant, zeros, ones, eye, isConstant } from './expr/index.js';

// === Expression Utilities ===
export { exprShape, exprVariables, isConstantExpr, resetExprIds } from './expr/index.js';

// === Affine Atoms ===
export {
  add,
  sub,
  neg,
  mul,
  div,
  matmul,
  sum,
  reshape,
  index,
  vstack,
  hstack,
  transpose,
  trace,
  diag,
  dot,
} from './atoms/index.js';

// === Nonlinear Atoms ===
export {
  norm1,
  norm2,
  normInf,
  norm,
  abs,
  pos,
  maximum,
  minimum,
  sumSquares,
  quadForm,
  quadOverLin,
} from './atoms/index.js';

// === DCP Analysis ===
export {
  Curvature,
  curvature,
  isConvex,
  isConcave,
  isAffine,
  isDcpConvex,
  isDcpConcave,
  isDcpAffine,
} from './dcp/index.js';

export {
  Sign,
  sign,
  isNonnegative,
  isNonpositive,
  exprIsNonnegative,
  exprIsNonpositive,
} from './dcp/index.js';

// === Sparse Matrix ===
export type { CscMatrix } from './sparse/index.js';
export {
  cscEmpty,
  cscIdentity,
  cscZeros,
  cscFromTriplets,
  cscFromDense,
  cscNnz,
  cscGet,
  cscScale,
  cscAdd,
  cscSub,
  cscVstack,
  cscHstack,
  cscTranspose,
  cscMulVec,
  cscMulMat,
  cscDiag,
  cscToDense,
  cscClone,
  cscEquals,
} from './sparse/index.js';

// === Constraints ===
export type { Constraint } from './constraints/index.js';
export { eq, le, ge, soc, constraintVariables, isDcpConstraint } from './constraints/index.js';

// === Problem ===
export type { ObjectiveSense, SolveStatus, Solution, SolverSettings } from './problem.js';
export { Problem } from './problem.js';

// === Solver ===
export { loadWasm, testWasm, clarabelVersion, solveConic } from './solver/index.js';

export type { ConicSolveResult } from './solver/index.js';

// === Errors ===
export {
  CvxError,
  DcpError,
  ShapeError,
  SolverError,
  InfeasibleError,
  UnboundedError,
} from './error.js';
