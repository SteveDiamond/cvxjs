// Shape utilities
export type { Shape } from './shape.js';
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
  broadcastShape,
  normalizeShape,
} from './shape.js';

// Core expression types
export type { ExprId, ArrayData, IndexRange, Expr } from './expression.js';
export {
  newExprId,
  resetExprIds,
  exprShape,
  exprVariables,
  isConstantExpr,
  arrayDataSize,
  isScalarData,
  getScalarValue,
} from './expression.js';

// Variable creation
export type { VariableOptions } from './variable.js';
export {
  VariableBuilder,
  variable,
  scalarVar,
  vectorVar,
  matrixVar,
  getVariableId,
  isVariable,
} from './variable.js';

// Constant creation
export {
  constant,
  constantFromData,
  toArrayData,
  scalarConst,
  vectorConst,
  matrixConst,
  zeros,
  ones,
  eye,
  isConstant,
  getConstantData,
  getScalarConstant,
} from './constant.js';

// Expression wrapper class
export { Expression, wrap, isExpression } from './expr-wrapper.js';
export type { ExprInput, ArrayInput } from './expr-wrapper.js';
