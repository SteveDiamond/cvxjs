import { Expr, ArrayData } from '../expr/index.js';

/**
 * Sign of an expression.
 *
 * Used for DCP composition rules where the sign of an argument
 * affects whether certain compositions are valid.
 */
export enum Sign {
  /** Expression is always >= 0 */
  Nonnegative = 'nonneg',
  /** Expression is always <= 0 */
  Nonpositive = 'nonpos',
  /** Expression is always = 0 */
  Zero = 'zero',
  /** Sign is unknown or mixed */
  Unknown = 'unknown',
}

/** Check if sign is nonnegative (includes zero) */
export function isNonnegative(s: Sign): boolean {
  return s === Sign.Nonnegative || s === Sign.Zero;
}

/** Check if sign is nonpositive (includes zero) */
export function isNonpositive(s: Sign): boolean {
  return s === Sign.Nonpositive || s === Sign.Zero;
}

/** Check if sign is zero */
export function isZero(s: Sign): boolean {
  return s === Sign.Zero;
}

/**
 * Combine signs under addition.
 */
export function addSign(a: Sign, b: Sign): Sign {
  if (a === Sign.Zero) return b;
  if (b === Sign.Zero) return a;
  if (a === b) return a;
  return Sign.Unknown;
}

/**
 * Negate a sign.
 */
export function negateSign(s: Sign): Sign {
  if (s === Sign.Nonnegative) return Sign.Nonpositive;
  if (s === Sign.Nonpositive) return Sign.Nonnegative;
  return s; // Zero and Unknown unchanged
}

/**
 * Multiply signs.
 */
export function mulSign(a: Sign, b: Sign): Sign {
  if (a === Sign.Zero || b === Sign.Zero) return Sign.Zero;
  if (a === Sign.Unknown || b === Sign.Unknown) return Sign.Unknown;

  // Same sign -> nonnegative
  if (a === b) return Sign.Nonnegative;

  // Different signs -> nonpositive
  return Sign.Nonpositive;
}

/**
 * Get the sign of ArrayData.
 */
export function arrayDataSign(data: ArrayData): Sign {
  if (data.type === 'scalar') {
    if (data.value === 0) return Sign.Zero;
    if (data.value > 0) return Sign.Nonnegative;
    return Sign.Nonpositive;
  }

  // For arrays, check all elements
  const values = data.type === 'dense' ? data.data : data.values;

  let hasPositive = false;
  let hasNegative = false;

  for (const v of values) {
    if (v > 0) hasPositive = true;
    if (v < 0) hasNegative = true;
  }

  if (!hasPositive && !hasNegative) return Sign.Zero;
  if (!hasNegative) return Sign.Nonnegative;
  if (!hasPositive) return Sign.Nonpositive;
  return Sign.Unknown;
}

/**
 * Compute the sign of an expression.
 */
export function sign(expr: Expr): Sign {
  switch (expr.kind) {
    case 'variable':
      if (expr.nonneg) return Sign.Nonnegative;
      if (expr.nonpos) return Sign.Nonpositive;
      return Sign.Unknown;

    case 'constant':
      return arrayDataSign(expr.value);

    case 'add':
      return addSign(sign(expr.left), sign(expr.right));

    case 'neg':
      return negateSign(sign(expr.arg));

    case 'mul': {
      return mulSign(sign(expr.left), sign(expr.right));
    }

    case 'div': {
      return mulSign(sign(expr.left), sign(expr.right));
    }

    case 'matmul': {
      // Matrix multiplication sign is complex - be conservative
      const leftSign = sign(expr.left);
      const rightSign = sign(expr.right);

      if (leftSign === Sign.Zero || rightSign === Sign.Zero) return Sign.Zero;
      if (isNonnegative(leftSign) && isNonnegative(rightSign)) return Sign.Nonnegative;
      if (isNonpositive(leftSign) && isNonpositive(rightSign)) return Sign.Nonnegative;
      if (isNonnegative(leftSign) && isNonpositive(rightSign)) return Sign.Nonpositive;
      if (isNonpositive(leftSign) && isNonnegative(rightSign)) return Sign.Nonpositive;
      return Sign.Unknown;
    }

    case 'sum':
    case 'trace':
    case 'cumsum':
      return sign(expr.arg);

    case 'reshape':
    case 'index':
    case 'transpose':
    case 'diag':
      return sign(expr.arg);

    case 'vstack':
    case 'hstack': {
      let result = Sign.Zero;
      for (const arg of expr.args) {
        result = addSign(result, sign(arg));
      }
      return result;
    }

    // Nonlinear atoms with known sign
    case 'norm1':
    case 'norm2':
    case 'normInf':
    case 'abs':
    case 'sumSquares':
    case 'quadForm':
    case 'quadOverLin':
    case 'exp': // e^x > 0 always
      return Sign.Nonnegative;

    case 'log':
      // log(x) can be any sign (log(x) < 0 for x < 1, log(x) > 0 for x > 1)
      return Sign.Unknown;

    case 'entropy':
      // -x*log(x) is nonnegative for 0 < x <= 1, nonpositive for x > 1
      return Sign.Unknown;

    case 'sqrt':
      // sqrt(x) >= 0 for x >= 0
      return Sign.Nonnegative;

    case 'power': {
      // x^p for x >= 0 is nonnegative for any p
      const argSign = sign(expr.arg);
      if (isNonnegative(argSign)) {
        return Sign.Nonnegative;
      }
      return Sign.Unknown;
    }

    case 'pos':
      return Sign.Nonnegative; // max(x, 0) >= 0

    case 'negPart':
      return Sign.Nonnegative; // max(-x, 0) >= 0

    case 'maximum': {
      // max is nonnegative if any argument is nonnegative
      for (const arg of expr.args) {
        if (isNonnegative(sign(arg))) return Sign.Nonnegative;
      }
      return Sign.Unknown;
    }

    case 'minimum': {
      // min is nonpositive if any argument is nonpositive
      for (const arg of expr.args) {
        if (isNonpositive(sign(arg))) return Sign.Nonpositive;
      }
      return Sign.Unknown;
    }

    default: {
      const _exhaustive: never = expr;
      return Sign.Unknown;
    }
  }
}

/**
 * Check if an expression is nonnegative.
 */
export function exprIsNonnegative(expr: Expr): boolean {
  return isNonnegative(sign(expr));
}

/**
 * Check if an expression is nonpositive.
 */
export function exprIsNonpositive(expr: Expr): boolean {
  return isNonpositive(sign(expr));
}
