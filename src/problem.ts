import { Expr, ExprId, exprVariables, exprShape } from './expr/index.js';
import { Constraint, isDcpConstraint, validateDcpConstraint, constraintVariables } from './constraints/index.js';
import { curvature, Curvature, isConvex, isConcave } from './dcp/index.js';
import { DcpError } from './error.js';
import { isScalar } from './expr/shape.js';

/**
 * Objective sense for optimization.
 */
export type ObjectiveSense = 'minimize' | 'maximize';

/**
 * Solve status from the solver.
 */
export type SolveStatus =
  | 'optimal'
  | 'infeasible'
  | 'unbounded'
  | 'max_iterations'
  | 'numerical_error'
  | 'unknown';

/**
 * Solution from solving a problem.
 */
export interface Solution {
  /** Solve status */
  readonly status: SolveStatus;
  /** Optimal objective value (if optimal) */
  readonly value?: number;
  /** Primal variable values (if optimal) */
  readonly primal?: ReadonlyMap<ExprId, Float64Array>;
  /** Solve time in seconds */
  readonly solveTime?: number;
  /** Number of iterations */
  readonly iterations?: number;
}

/**
 * Solver settings.
 */
export interface SolverSettings {
  /** Print solver output */
  verbose?: boolean;
  /** Maximum iterations */
  maxIter?: number;
  /** Time limit in seconds */
  timeLimit?: number;
  /** Absolute gap tolerance */
  tolGapAbs?: number;
  /** Relative gap tolerance */
  tolGapRel?: number;
}

/**
 * Problem builder for constructing and solving optimization problems.
 *
 * @example
 * ```ts
 * const x = variable(5);
 *
 * const solution = await Problem.minimize(sum(x))
 *   .subjectTo([ge(x, 1)])
 *   .solve();
 *
 * console.log('Optimal value:', solution.value);
 * ```
 */
export class Problem {
  private readonly _objective: Expr;
  private readonly _sense: ObjectiveSense;
  private _constraints: Constraint[] = [];
  private _settings: SolverSettings = {};

  private constructor(objective: Expr, sense: ObjectiveSense) {
    // Validate objective is scalar
    const objShape = exprShape(objective);
    if (!isScalar(objShape)) {
      throw new DcpError(`Objective must be scalar, got shape (${objShape.dims.join(', ')})`);
    }

    this._objective = objective;
    this._sense = sense;
  }

  /**
   * Create a minimization problem.
   *
   * @example
   * ```ts
   * Problem.minimize(norm2(x))
   *   .subjectTo([ge(x, 0)])
   *   .solve();
   * ```
   */
  static minimize(objective: Expr): Problem {
    return new Problem(objective, 'minimize');
  }

  /**
   * Create a maximization problem.
   *
   * @example
   * ```ts
   * Problem.maximize(sum(log(x)))
   *   .subjectTo([le(sum(x), 1)])
   *   .solve();
   * ```
   */
  static maximize(objective: Expr): Problem {
    return new Problem(objective, 'maximize');
  }

  /**
   * Add constraints to the problem.
   *
   * @example
   * ```ts
   * problem.subjectTo([
   *   eq(sum(x), 1),
   *   ge(x, 0),
   * ])
   * ```
   */
  subjectTo(constraints: Constraint[]): this {
    this._constraints = [...this._constraints, ...constraints];
    return this;
  }

  /**
   * Set solver settings.
   *
   * @example
   * ```ts
   * problem.settings({ verbose: true, maxIter: 1000 })
   * ```
   */
  settings(settings: SolverSettings): this {
    this._settings = { ...this._settings, ...settings };
    return this;
  }

  /**
   * Get the objective expression.
   */
  get objective(): Expr {
    return this._objective;
  }

  /**
   * Get the optimization sense.
   */
  get sense(): ObjectiveSense {
    return this._sense;
  }

  /**
   * Get the constraints.
   */
  get constraints(): readonly Constraint[] {
    return this._constraints;
  }

  /**
   * Get all variables in the problem.
   */
  get variables(): Set<ExprId> {
    const vars = exprVariables(this._objective);
    for (const c of this._constraints) {
      for (const v of constraintVariables(c)) {
        vars.add(v);
      }
    }
    return vars;
  }

  /**
   * Check if the problem satisfies DCP rules.
   */
  isDcp(): boolean {
    // Check objective curvature
    const objCurv = curvature(this._objective);

    if (this._sense === 'minimize' && !isConvex(objCurv)) {
      return false;
    }
    if (this._sense === 'maximize' && !isConcave(objCurv)) {
      return false;
    }

    // Check all constraints
    for (const c of this._constraints) {
      if (!isDcpConstraint(c)) {
        return false;
      }
    }

    return true;
  }

  /**
   * Validate DCP compliance and throw if not valid.
   */
  validateDcp(): void {
    const objCurv = curvature(this._objective);

    if (this._sense === 'minimize' && !isConvex(objCurv)) {
      throw new DcpError(
        `Minimization objective must be convex, got ${objCurv}`
      );
    }
    if (this._sense === 'maximize' && !isConcave(objCurv)) {
      throw new DcpError(
        `Maximization objective must be concave, got ${objCurv}`
      );
    }

    for (const c of this._constraints) {
      validateDcpConstraint(c);
    }
  }

  /**
   * Solve the optimization problem.
   *
   * @returns Solution containing status, optimal value, and variable values
   * @throws DcpError if problem is not DCP compliant
   *
   * @example
   * ```ts
   * const solution = await problem.solve();
   * if (solution.status === 'optimal') {
   *   console.log('Optimal value:', solution.value);
   * }
   * ```
   */
  async solve(): Promise<Solution> {
    // Validate DCP
    this.validateDcp();

    // TODO: Implement canonicalization and solver call
    // For now, return a placeholder
    throw new Error('Solver not yet implemented. Coming in Phase 4-5.');
  }
}
