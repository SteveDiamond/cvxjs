import { Expr, exprShape, exprVariables, ExprId } from '../expr/index.js';
import { curvature, isConcave, isAffine } from '../dcp/index.js';
import { sub, toExpr } from '../atoms/affine.js';
import { ShapeError, DcpError } from '../error.js';
import { shapeToString, broadcastShape } from '../expr/shape.js';

/**
 * Constraint types for optimization problems.
 *
 * - Zero: Equality constraint (expr == 0)
 * - NonNeg: Inequality constraint (expr >= 0)
 * - SOC: Second-order cone constraint (||x||_2 <= t)
 */
export type Constraint =
  | { readonly kind: 'zero'; readonly expr: Expr }
  | { readonly kind: 'nonneg'; readonly expr: Expr }
  | { readonly kind: 'soc'; readonly t: Expr; readonly x: Expr };

/**
 * Create an equality constraint: lhs == rhs
 *
 * @example
 * ```ts
 * eq(sum(x), 1)  // sum(x) == 1
 * ```
 */
export function eq(lhs: Expr | number, rhs: Expr | number): Constraint {
  const l = toExpr(lhs);
  const r = toExpr(rhs);

  // Validate shapes are broadcastable
  const lShape = exprShape(l);
  const rShape = exprShape(r);
  if (!broadcastShape(lShape, rShape)) {
    throw new ShapeError(
      'Cannot create equality constraint with incompatible shapes',
      shapeToString(lShape),
      shapeToString(rShape)
    );
  }

  return { kind: 'zero', expr: sub(l, r) };
}

/**
 * Create an inequality constraint: lhs <= rhs
 *
 * @example
 * ```ts
 * le(x, 10)     // x <= 10
 * le(norm2(x), 1)  // ||x||_2 <= 1
 * ```
 */
export function le(lhs: Expr | number, rhs: Expr | number): Constraint {
  const l = toExpr(lhs);
  const r = toExpr(rhs);

  // Validate shapes are broadcastable
  const lShape = exprShape(l);
  const rShape = exprShape(r);
  if (!broadcastShape(lShape, rShape)) {
    throw new ShapeError(
      'Cannot create inequality constraint with incompatible shapes',
      shapeToString(lShape),
      shapeToString(rShape)
    );
  }

  // lhs <= rhs  <==>  rhs - lhs >= 0
  return { kind: 'nonneg', expr: sub(r, l) };
}

/**
 * Create an inequality constraint: lhs >= rhs
 *
 * @example
 * ```ts
 * ge(x, 0)      // x >= 0
 * ge(sum(x), 1)  // sum(x) >= 1
 * ```
 */
export function ge(lhs: Expr | number, rhs: Expr | number): Constraint {
  const l = toExpr(lhs);
  const r = toExpr(rhs);

  // Validate shapes are broadcastable
  const lShape = exprShape(l);
  const rShape = exprShape(r);
  if (!broadcastShape(lShape, rShape)) {
    throw new ShapeError(
      'Cannot create inequality constraint with incompatible shapes',
      shapeToString(lShape),
      shapeToString(rShape)
    );
  }

  // lhs >= rhs  <==>  lhs - rhs >= 0
  return { kind: 'nonneg', expr: sub(l, r) };
}

/**
 * Create a second-order cone constraint: ||x||_2 <= t
 *
 * @example
 * ```ts
 * soc(norm2(x), t)  // ||x||_2 <= t
 * ```
 */
export function soc(x: Expr, t: Expr | number): Constraint {
  return { kind: 'soc', t: toExpr(t), x };
}

/**
 * Get all variables referenced in a constraint.
 */
export function constraintVariables(constraint: Constraint): Set<ExprId> {
  switch (constraint.kind) {
    case 'zero':
    case 'nonneg':
      return exprVariables(constraint.expr);
    case 'soc': {
      const vars = exprVariables(constraint.t);
      for (const v of exprVariables(constraint.x)) {
        vars.add(v);
      }
      return vars;
    }
  }
}

/**
 * Check if a constraint satisfies DCP rules.
 *
 * DCP rules for constraints:
 * - Zero (equality): expression must be affine
 * - NonNeg (inequality): must be convex <= concave form
 *   Since we store as (rhs - lhs >= 0), expr must be concave
 *   But more generally: original lhs must be convex, rhs must be concave
 * - SOC: t and x must be affine
 */
export function isDcpConstraint(constraint: Constraint): boolean {
  switch (constraint.kind) {
    case 'zero':
      // Equality requires affine expression
      return isAffine(curvature(constraint.expr));

    case 'nonneg': {
      // For lhs <= rhs, we have expr = rhs - lhs
      // DCP requires: lhs convex, rhs concave
      // Since we store rhs - lhs >= 0:
      // If lhs is convex and rhs is concave, then rhs - lhs is concave
      // So expr should be concave (includes affine)
      const curv = curvature(constraint.expr);
      return isConcave(curv);
    }

    case 'soc':
      // SOC requires affine arguments
      return isAffine(curvature(constraint.t)) && isAffine(curvature(constraint.x));
  }
}

/**
 * Validate that a constraint satisfies DCP rules.
 * Throws DcpError if not valid.
 */
export function validateDcpConstraint(constraint: Constraint): void {
  if (!isDcpConstraint(constraint)) {
    switch (constraint.kind) {
      case 'zero':
        throw new DcpError(
          `Equality constraint requires affine expression, got ${curvature(constraint.expr)}`
        );
      case 'nonneg':
        throw new DcpError(
          `Inequality constraint requires convex <= concave form, got curvature ${curvature(constraint.expr)}`
        );
      case 'soc':
        throw new DcpError(
          `SOC constraint requires affine arguments, got t: ${curvature(constraint.t)}, x: ${curvature(constraint.x)}`
        );
    }
  }
}
