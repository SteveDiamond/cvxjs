/**
 * cvxjs - Disciplined Convex Programming in TypeScript
 *
 * @example
 * ```ts
 * import { variable, constant, Problem } from 'cvxjs';
 *
 * // Create variables and constants
 * const x = variable(5);
 * const A = constant([[1, 2, 3, 4, 5]]);
 *
 * // Build expressions with fluent method chaining
 * const residual = A.matmul(x).sub(10);
 * const objective = residual.norm2().add(x.norm1().mul(0.1));
 *
 * // Solve with fluent constraints
 * const solution = await Problem.minimize(objective)
 *   .subjectTo([x.ge(0), x.sum().le(10)])
 *   .solve();
 *
 * // Access results
 * console.log('Optimal value:', solution.value);
 * console.log('x =', solution.valueOf(x));
 * ```
 *
 * @packageDocumentation
 */

// === Expression Types ===
export type { Shape, ExprId, ArrayData, IndexRange, ExprData } from './expr/index.js';

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

// === Expression Wrapper ===
export { Expr, wrap, isExpr } from './expr/index.js';
export type { ExprInput, ArrayInput } from './expr/index.js';

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
  cumsum,
} from './atoms/index.js';

// === Nonlinear Atoms ===
export {
  norm1,
  norm2,
  normInf,
  norm,
  abs,
  pos,
  negPart,
  maximum,
  minimum,
  sumSquares,
  quadForm,
  quadOverLin,
  exp,
  log,
  entropy,
  sqrt,
  power,
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
