import { ExprData, ExprId, newExprId } from './expr-data.js';
import { Shape, normalizeShape, vector, matrix, scalar } from './shape.js';
import { Expr } from './expr.js';

/**
 * Options for variable creation.
 */
export interface VariableOptions {
  /** Optional name for debugging */
  name?: string;
  /** Constrain variable to be >= 0 */
  nonneg?: boolean;
  /** Constrain variable to be <= 0 */
  nonpos?: boolean;
  /** Constrain variable to integer values (requires HiGHS solver) */
  integer?: boolean;
  /** Constrain variable to binary {0, 1} values (requires HiGHS solver) */
  binary?: boolean;
}

/**
 * Builder for creating optimization variables.
 *
 * @example
 * ```ts
 * const x = new VariableBuilder(5).name('x').nonneg().build();
 * const A = new VariableBuilder([3, 4]).name('A').build();
 * ```
 */
export class VariableBuilder {
  private readonly _shape: Shape;
  private _name?: string;
  private _nonneg = false;
  private _nonpos = false;
  private _integer = false;
  private _binary = false;

  constructor(shape: number | readonly [number] | readonly [number, number]) {
    this._shape = normalizeShape(shape);
  }

  /** Set the variable name */
  name(n: string): this {
    this._name = n;
    return this;
  }

  /** Constrain variable to be non-negative (>= 0) */
  nonneg(): this {
    this._nonneg = true;
    this._nonpos = false; // Mutually exclusive
    return this;
  }

  /** Constrain variable to be non-positive (<= 0) */
  nonpos(): this {
    this._nonpos = true;
    this._nonneg = false; // Mutually exclusive
    return this;
  }

  /** Constrain variable to integer values (requires HiGHS solver) */
  integer(): this {
    this._integer = true;
    this._binary = false; // Mutually exclusive
    return this;
  }

  /** Constrain variable to binary {0, 1} values (requires HiGHS solver) */
  binary(): this {
    this._binary = true;
    this._integer = false; // Mutually exclusive
    // Binary implies nonneg bounds
    this._nonneg = true;
    this._nonpos = false;
    return this;
  }

  /** Build the variable expression */
  build(): Expr {
    return new Expr({
      kind: 'variable',
      id: newExprId(),
      shape: this._shape,
      name: this._name,
      nonneg: this._nonneg || undefined,
      nonpos: this._nonpos || undefined,
      integer: this._integer || undefined,
      binary: this._binary || undefined,
    });
  }
}

/**
 * Create an optimization variable.
 *
 * @param shape - Size as number (vector) or [rows, cols] (matrix)
 * @param options - Optional name and constraints
 *
 * @example
 * ```ts
 * const x = variable(5);                    // 5-element vector
 * const A = variable([3, 4]);               // 3x4 matrix
 * const y = variable(3, { nonneg: true });  // Non-negative vector
 * ```
 */
export function variable(
  shape: number | readonly [number] | readonly [number, number],
  options?: VariableOptions
): Expr {
  const builder = new VariableBuilder(shape);
  if (options?.name) builder.name(options.name);
  if (options?.nonneg) builder.nonneg();
  if (options?.nonpos) builder.nonpos();
  if (options?.integer) builder.integer();
  if (options?.binary) builder.binary();
  return builder.build();
}

/**
 * Create a scalar optimization variable.
 *
 * @example
 * ```ts
 * const t = scalarVar();
 * const t = scalarVar({ name: 't', nonneg: true });
 * ```
 */
export function scalarVar(options?: VariableOptions): Expr {
  return new Expr({
    kind: 'variable',
    id: newExprId(),
    shape: scalar(),
    name: options?.name,
    nonneg: options?.nonneg || options?.binary,
    nonpos: options?.nonpos,
    integer: options?.integer,
    binary: options?.binary,
  });
}

/**
 * Create a vector optimization variable.
 *
 * @example
 * ```ts
 * const x = vectorVar(5);
 * const x = vectorVar(5, { name: 'x' });
 * ```
 */
export function vectorVar(n: number, options?: VariableOptions): Expr {
  return new Expr({
    kind: 'variable',
    id: newExprId(),
    shape: vector(n),
    name: options?.name,
    nonneg: options?.nonneg || options?.binary,
    nonpos: options?.nonpos,
    integer: options?.integer,
    binary: options?.binary,
  });
}

/**
 * Create a matrix optimization variable.
 *
 * @example
 * ```ts
 * const X = matrixVar(3, 4);
 * const X = matrixVar(3, 4, { name: 'X' });
 * ```
 */
export function matrixVar(rows: number, cols: number, options?: VariableOptions): Expr {
  return new Expr({
    kind: 'variable',
    id: newExprId(),
    shape: matrix(rows, cols),
    name: options?.name,
    nonneg: options?.nonneg || options?.binary,
    nonpos: options?.nonpos,
    integer: options?.integer,
    binary: options?.binary,
  });
}

/**
 * Get the ID of a variable expression.
 * Throws if the expression is not a variable.
 */
export function getVariableId(expr: ExprData | Expr): ExprId {
  const e = expr instanceof Expr ? expr.data : expr;
  if (e.kind !== 'variable') {
    throw new Error(`Expected variable, got ${e.kind}`);
  }
  return e.id;
}

/**
 * Check if an expression is a variable.
 */
export function isVariable(expr: ExprData | Expr): boolean {
  const e = expr instanceof Expr ? expr.data : expr;
  return e.kind === 'variable';
}
