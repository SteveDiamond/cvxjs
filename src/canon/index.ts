export type { LinExpr } from './lin-expr.js';
export {
  linExprVariable,
  linExprConstant,
  linExprZero,
  linExprAdd,
  linExprSub,
  linExprNeg,
  linExprScale,
  linExprLeftMul,
  linExprSum,
  linExprVstack,
  linExprIsConstant,
  linExprGetConstant,
} from './lin-expr.js';

export type { QuadExpr } from './quad-expr.js';
export {
  quadCoeffKey,
  parseQuadCoeffKey,
  quadExprFromLinear,
  quadExprQuadratic,
  quadExprSumSquares,
  quadExprIsLinear,
  quadExprAdd,
  quadExprScale,
  quadExprVariables,
} from './quad-expr.js';

export type { ConeConstraint, ConeDims } from './cone-constraint.js';
export { emptyConeDims, coneDimsRows } from './cone-constraint.js';

export type { AuxVar, CanonResult, CanonExpr } from './canonicalizer.js';
export {
  Canonicalizer,
  canonicalizeProblem,
  canonExprAsLinear,
  canonExprToQuadratic,
} from './canonicalizer.js';

export type { VariableMap, StuffedProblem } from './stuffing.js';
export {
  buildVariableMap,
  stuffLinExpr,
  stuffObjective,
  stuffQuadraticObjective,
  stuffProblem,
} from './stuffing.js';
