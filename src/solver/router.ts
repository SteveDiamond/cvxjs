/**
 * Solver routing logic.
 *
 * Analyzes problems to determine whether to use HiGHS (for MIP/LP)
 * or Clarabel (for conic optimization).
 */

import type { ExprData, ExprId } from '../expr/index.js';
import type { Constraint } from '../constraints/index.js';
import type { ConeConstraint, ConeDims } from '../canon/cone-constraint.js';
import { DcpError } from '../error.js';

/**
 * Variable properties extracted from expressions.
 */
export interface VariableProps {
  readonly nonneg?: boolean;
  readonly nonpos?: boolean;
  readonly integer?: boolean;
  readonly binary?: boolean;
}

/**
 * Analysis result for a problem.
 */
export interface ProblemAnalysis {
  /** True if problem contains integer variables */
  hasIntegerVars: boolean;
  /** True if problem contains binary variables */
  hasBinaryVars: boolean;
  /** True if problem requires conic constraints (SOC, exp, power) */
  hasConicConstraints: boolean;
}

/**
 * Solver type selection.
 */
export type SolverType = 'clarabel' | 'highs';

/**
 * Collect variable properties from an expression tree.
 */
export function collectVariableProps(
  expr: ExprData,
  props: Map<ExprId, VariableProps>
): void {
  if (expr.kind === 'variable') {
    props.set(expr.id, {
      nonneg: expr.nonneg,
      nonpos: expr.nonpos,
      integer: expr.integer,
      binary: expr.binary,
    });
    return;
  }

  // Recursively process sub-expressions
  switch (expr.kind) {
    case 'constant':
      break;
    case 'add':
    case 'mul':
    case 'div':
    case 'matmul':
      collectVariableProps(expr.left, props);
      collectVariableProps(expr.right, props);
      break;
    case 'neg':
    case 'sum':
    case 'reshape':
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
      collectVariableProps(expr.arg, props);
      break;
    case 'index':
      collectVariableProps(expr.arg, props);
      break;
    case 'vstack':
    case 'hstack':
    case 'maximum':
    case 'minimum':
      for (const arg of expr.args) {
        collectVariableProps(arg, props);
      }
      break;
    case 'quadForm':
      collectVariableProps(expr.x, props);
      collectVariableProps(expr.P, props);
      break;
    case 'quadOverLin':
      collectVariableProps(expr.x, props);
      collectVariableProps(expr.y, props);
      break;
  }
}

/**
 * Collect variable properties from a constraint.
 */
export function collectVariablePropsFromConstraint(
  constraint: Constraint,
  props: Map<ExprId, VariableProps>
): void {
  switch (constraint.kind) {
    case 'zero':
    case 'nonneg':
      collectVariableProps(constraint.expr, props);
      break;
    case 'soc':
      collectVariableProps(constraint.t, props);
      collectVariableProps(constraint.x, props);
      break;
  }
}

/**
 * Check if cone dimensions contain non-linear conic constraints.
 */
export function hasNonLinearCones(coneDims: ConeDims): boolean {
  return coneDims.soc.length > 0 || coneDims.exp > 0 || coneDims.power.length > 0;
}

/**
 * Check if canonicalized constraints contain non-linear conic constraints.
 */
export function hasConicConstraintsInList(constraints: ConeConstraint[]): boolean {
  for (const c of constraints) {
    if (c.kind === 'soc' || c.kind === 'exp' || c.kind === 'power') {
      return true;
    }
  }
  return false;
}

/**
 * Analyze a problem to determine its characteristics.
 */
export function analyzeProblem(
  varProps: Map<ExprId, VariableProps>,
  coneDims: ConeDims
): ProblemAnalysis {
  let hasIntegerVars = false;
  let hasBinaryVars = false;

  for (const props of varProps.values()) {
    if (props.integer) hasIntegerVars = true;
    if (props.binary) hasBinaryVars = true;
  }

  return {
    hasIntegerVars,
    hasBinaryVars,
    hasConicConstraints: hasNonLinearCones(coneDims),
  };
}

/**
 * Analyze a problem before canonicalization to check for integer variables.
 * Used for early error detection.
 */
export function analyzePreCanon(varProps: Map<ExprId, VariableProps>): {
  hasIntegerVars: boolean;
  hasBinaryVars: boolean;
} {
  let hasIntegerVars = false;
  let hasBinaryVars = false;

  for (const props of varProps.values()) {
    if (props.integer) hasIntegerVars = true;
    if (props.binary) hasBinaryVars = true;
  }

  return { hasIntegerVars, hasBinaryVars };
}

/**
 * Select the appropriate solver for a problem.
 *
 * @throws DcpError if the problem has both integer variables and conic constraints
 */
export function selectSolver(analysis: ProblemAnalysis): SolverType {
  const hasIntegers = analysis.hasIntegerVars || analysis.hasBinaryVars;

  // Error if both integers and conic constraints
  if (hasIntegers && analysis.hasConicConstraints) {
    throw new DcpError(
      'Problems with integer/binary variables cannot have conic constraints (SOC, exp, power). ' +
        'Mixed-integer conic programming is not supported.'
    );
  }

  // Use HiGHS for integer problems
  if (hasIntegers) {
    return 'highs';
  }

  // Default to Clarabel for all other problems
  return 'clarabel';
}
