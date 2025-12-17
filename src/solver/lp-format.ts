/**
 * CPLEX LP format generator.
 *
 * Converts canonical optimization problems to LP format strings
 * that can be solved by HiGHS.
 */

import type { ExprId } from '../expr/index.js';
import type { LinExpr } from '../canon/lin-expr.js';
import type { QuadExpr } from '../canon/quad-expr.js';
import type { ConeConstraint } from '../canon/cone-constraint.js';
import type { VariableMap } from '../canon/stuffing.js';
import type { VariableProps } from './router.js';
import type { ObjectiveSense } from '../problem.js';

/**
 * Variable info for LP format generation.
 */
export interface LPVariableInfo {
  readonly name: string;
  readonly nonneg?: boolean;
  readonly nonpos?: boolean;
  readonly integer?: boolean;
  readonly binary?: boolean;
}

/**
 * Result of LP format generation.
 */
export interface LPFormatResult {
  /** LP format string */
  readonly lpString: string;
  /** Ordered variable names (for solution extraction) */
  readonly varNames: string[];
}

/**
 * Format a coefficient for LP format.
 * Returns string like "+ 3.5 x_0" or "- 2 x_1"
 */
function formatCoeff(coeff: number, varName: string, isFirst: boolean): string {
  if (coeff === 0) return '';

  const sign = coeff >= 0 ? '+' : '-';
  const absCoeff = Math.abs(coeff);

  // For first term, don't include leading +
  const signStr = isFirst ? (coeff < 0 ? '- ' : '') : ` ${sign} `;

  if (absCoeff === 1) {
    return `${signStr}${varName}`;
  }
  return `${signStr}${absCoeff} ${varName}`;
}

/**
 * Format a quadratic coefficient for LP format.
 * Returns string like "+ 3.5 x_0 * x_1"
 */
function formatQuadCoeff(
  coeff: number,
  varName1: string,
  varName2: string,
  isFirst: boolean
): string {
  if (coeff === 0) return '';

  const sign = coeff >= 0 ? '+' : '-';
  const absCoeff = Math.abs(coeff);

  const signStr = isFirst ? (coeff < 0 ? '- ' : '') : ` ${sign} `;

  if (varName1 === varName2) {
    // Diagonal term: x^2
    if (absCoeff === 1) {
      return `${signStr}${varName1} ^ 2`;
    }
    return `${signStr}${absCoeff} ${varName1} ^ 2`;
  } else {
    // Off-diagonal term: x * y
    if (absCoeff === 1) {
      return `${signStr}${varName1} * ${varName2}`;
    }
    return `${signStr}${absCoeff} ${varName1} * ${varName2}`;
  }
}

/**
 * Convert a linear expression to LP format terms.
 */
function linExprToTerms(
  linExpr: LinExpr,
  varMap: VariableMap,
  varNames: string[],
  rowIndex: number
): string {
  const terms: string[] = [];
  let isFirst = true;

  // Iterate through coefficient matrices
  for (const [varId, coeffMatrix] of linExpr.coeffs) {
    const mapping = varMap.idToCol.get(varId);
    if (!mapping) continue;

    // Extract coefficients for this row
    for (let col = 0; col < coeffMatrix.ncols; col++) {
      for (let k = coeffMatrix.colPtr[col]!; k < coeffMatrix.colPtr[col + 1]!; k++) {
        if (coeffMatrix.rowIdx[k] === rowIndex) {
          const coeff = coeffMatrix.values[k]!;
          const globalCol = mapping.start + col;
          const name = varNames[globalCol]!;
          const term = formatCoeff(coeff, name, isFirst);
          if (term) {
            terms.push(term);
            isFirst = false;
          }
        }
      }
    }
  }

  return terms.join('') || '0';
}

/**
 * Convert quadratic expression to LP format objective.
 */
function quadExprToObjective(
  quadExpr: QuadExpr,
  varMap: VariableMap,
  varNames: string[]
): string {
  const terms: string[] = [];
  let isFirst = true;

  // Linear terms from the linear part
  if (quadExpr.linear.rows === 1) {
    for (const [varId, coeffMatrix] of quadExpr.linear.coeffs) {
      const mapping = varMap.idToCol.get(varId);
      if (!mapping) continue;

      for (let col = 0; col < coeffMatrix.ncols; col++) {
        for (let k = coeffMatrix.colPtr[col]!; k < coeffMatrix.colPtr[col + 1]!; k++) {
          if (coeffMatrix.rowIdx[k] === 0) {
            const coeff = coeffMatrix.values[k]!;
            const globalCol = mapping.start + col;
            const name = varNames[globalCol]!;
            const term = formatCoeff(coeff, name, isFirst);
            if (term) {
              terms.push(term);
              isFirst = false;
            }
          }
        }
      }
    }
  }

  // Quadratic terms
  // quadCoeffs maps "varId1,varId2" -> CscMatrix
  const quadTerms: string[] = [];
  let quadFirst = true;

  for (const [key, coeffMatrix] of quadExpr.quadCoeffs) {
    // Parse the key to get variable IDs
    const [id1Str, id2Str] = key.split(',');
    const id1 = parseInt(id1Str!, 10) as ExprId;
    const id2 = parseInt(id2Str!, 10) as ExprId;

    const mapping1 = varMap.idToCol.get(id1);
    const mapping2 = varMap.idToCol.get(id2);
    if (!mapping1 || !mapping2) continue;

    // coeffMatrix is size1 x size2
    for (let c = 0; c < coeffMatrix.ncols; c++) {
      for (let k = coeffMatrix.colPtr[c]!; k < coeffMatrix.colPtr[c + 1]!; k++) {
        const r = coeffMatrix.rowIdx[k]!;
        const coeff = coeffMatrix.values[k]!;

        const globalCol1 = mapping1.start + r;
        const globalCol2 = mapping2.start + c;
        const name1 = varNames[globalCol1]!;
        const name2 = varNames[globalCol2]!;

        const term = formatQuadCoeff(coeff, name1, name2, quadFirst);
        if (term) {
          quadTerms.push(term);
          quadFirst = false;
        }
      }
    }
  }

  // Combine: linear + [ quadratic ] / 2
  let result = terms.join('') || '0';
  if (quadTerms.length > 0) {
    result += ` + [ ${quadTerms.join('')} ] / 2`;
  }

  return result;
}

/**
 * Generate LP format string from canonical problem.
 *
 * Note: The objective has already been canonicalized (negated for maximization),
 * so we always write "Minimize" in LP format. The caller handles adjusting
 * the final objective value.
 */
export function generateLP(
  objective: LinExpr,
  constraints: ConeConstraint[],
  varMap: VariableMap,
  varProps: Map<ExprId, VariableProps>,
  _sense: ObjectiveSense, // Unused - objective already canonicalized
  quadObjective?: QuadExpr
): LPFormatResult {
  const lines: string[] = [];

  // Build ordered variable names
  const varNames: string[] = new Array(varMap.totalVars);
  for (const [varId, mapping] of varMap.idToCol) {
    for (let i = 0; i < mapping.size; i++) {
      varNames[mapping.start + i] = `x_${varId}_${i}`;
    }
  }

  // Objective - always minimize since canonicalization handles sense
  lines.push('Minimize');

  if (quadObjective && quadObjective.quadCoeffs.size > 0) {
    // Quadratic objective
    lines.push(`  obj: ${quadExprToObjective(quadObjective, varMap, varNames)}`);
  } else {
    // Linear objective
    const objTerms = objective.rows === 1 ? linExprToTerms(objective, varMap, varNames, 0) : '0';
    lines.push(`  obj: ${objTerms}`);
  }

  // Constraints
  lines.push('Subject To');

  let constraintIdx = 0;
  for (const constraint of constraints) {
    if (constraint.kind === 'zero') {
      // Equality constraint: Ax + b = 0  =>  Ax = -b
      for (let row = 0; row < constraint.a.rows; row++) {
        const lhs = linExprToTerms(constraint.a, varMap, varNames, row);
        const rhs = -constraint.a.constant[row]!;
        lines.push(`  c${constraintIdx}: ${lhs} = ${rhs}`);
        constraintIdx++;
      }
    } else if (constraint.kind === 'nonneg') {
      // Inequality constraint: Ax + b >= 0  =>  Ax >= -b
      for (let row = 0; row < constraint.a.rows; row++) {
        const lhs = linExprToTerms(constraint.a, varMap, varNames, row);
        const rhs = -constraint.a.constant[row]!;
        lines.push(`  c${constraintIdx}: ${lhs} >= ${rhs}`);
        constraintIdx++;
      }
    } else {
      // SOC, exp, power constraints should not reach here
      throw new Error(`Unsupported constraint type for LP format: ${constraint.kind}`);
    }
  }

  // Bounds
  lines.push('Bounds');

  // Collect variable bounds
  for (const [varId, mapping] of varMap.idToCol) {
    const props = varProps.get(varId);
    for (let i = 0; i < mapping.size; i++) {
      const name = varNames[mapping.start + i]!;

      if (props?.binary) {
        // Binary: 0 <= x <= 1 (handled in Binary section)
        lines.push(`  0 <= ${name} <= 1`);
      } else if (props?.nonneg) {
        lines.push(`  0 <= ${name} <= +inf`);
      } else if (props?.nonpos) {
        lines.push(`  -inf <= ${name} <= 0`);
      } else {
        lines.push(`  ${name} free`);
      }
    }
  }

  // Integer variables (General section)
  const integerVars: string[] = [];
  for (const [varId, mapping] of varMap.idToCol) {
    const props = varProps.get(varId);
    if (props?.integer && !props?.binary) {
      for (let i = 0; i < mapping.size; i++) {
        integerVars.push(varNames[mapping.start + i]!);
      }
    }
  }

  if (integerVars.length > 0) {
    lines.push('General');
    for (const name of integerVars) {
      lines.push(`  ${name}`);
    }
  }

  // Binary variables
  const binaryVars: string[] = [];
  for (const [varId, mapping] of varMap.idToCol) {
    const props = varProps.get(varId);
    if (props?.binary) {
      for (let i = 0; i < mapping.size; i++) {
        binaryVars.push(varNames[mapping.start + i]!);
      }
    }
  }

  if (binaryVars.length > 0) {
    lines.push('Binary');
    for (const name of binaryVars) {
      lines.push(`  ${name}`);
    }
  }

  lines.push('End');

  return {
    lpString: lines.join('\n'),
    varNames,
  };
}
