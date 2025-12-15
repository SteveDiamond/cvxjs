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

export type { ConeConstraint, ConeDims } from './cone-constraint.js';
export { emptyConeDims, coneDimsRows } from './cone-constraint.js';

export type { AuxVar, CanonResult } from './canonicalizer.js';
export { Canonicalizer, canonicalizeProblem } from './canonicalizer.js';

export type { VariableMap, StuffedProblem } from './stuffing.js';
export { buildVariableMap, stuffLinExpr, stuffObjective, stuffProblem } from './stuffing.js';
