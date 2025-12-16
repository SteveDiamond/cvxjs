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
export type { ExprId, ArrayData, IndexRange, ExprData } from './expr-data.js';
export { newExprId, resetExprIds, arrayDataSize, isScalarData, getScalarValue } from './expr-data.js';
import {
  exprShape as exprShapeRaw,
  exprVariables as exprVariablesRaw,
  isConstantExpr as isConstantExprRaw,
} from './expr-data.js';
import type { ExprData, ExprId } from './expr-data.js';
import type { Shape } from './shape.js';
import { Expr } from './expr.js';

/** Get the shape of an expression (accepts both ExprData and Expr wrapper) */
export function exprShape(expr: ExprData | Expr): Shape {
  const data = expr instanceof Expr ? expr.data : expr;
  return exprShapeRaw(data);
}

/** Get all variable IDs referenced in an expression (accepts both ExprData and Expr wrapper) */
export function exprVariables(expr: ExprData | Expr): Set<ExprId> {
  const data = expr instanceof Expr ? expr.data : expr;
  return exprVariablesRaw(data);
}

/** Check if an expression is constant (accepts both ExprData and Expr wrapper) */
export function isConstantExpr(expr: ExprData | Expr): boolean {
  const data = expr instanceof Expr ? expr.data : expr;
  return isConstantExprRaw(data);
}

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
export { Expr, wrap, isExpr } from './expr.js';
export type { ExprInput, ArrayInput } from './expr.js';
