import { Expr, ExprId, exprVariables, exprShape, isVariable, getVariableId } from './expr/index.js';
import { size } from './expr/shape.js';
import { Constraint, isDcpConstraint, validateDcpConstraint, constraintVariables } from './constraints/index.js';
import { curvature, Curvature, isConvex, isConcave } from './dcp/index.js';
import { DcpError, SolverError, InfeasibleError, UnboundedError } from './error.js';
import { isScalar } from './expr/shape.js';
import { canonicalizeProblem, buildVariableMap, stuffProblem } from './canon/index.js';
import { solveConic } from './solver/index.js';

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
   * @throws SolverError if solver encounters an error
   * @throws InfeasibleError if problem is infeasible
   * @throws UnboundedError if problem is unbounded
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

    // Collect all variables and their sizes
    const varIds = this.variables;
    const varSizes = new Map<ExprId, number>();

    // Extract variable sizes from objective and constraints
    this.collectVariableSizes(this._objective, varSizes);
    for (const c of this._constraints) {
      this.collectVariableSizesFromConstraint(c, varSizes);
    }

    // Canonicalize the problem
    const { objectiveLinExpr, coneConstraints, auxVars, objectiveOffset } =
      canonicalizeProblem(this._objective, this._constraints, this._sense);

    // Build variable mapping
    const varMap = buildVariableMap(varIds, varSizes, auxVars);

    // Stuff the problem into standard form
    const stuffed = stuffProblem(
      objectiveLinExpr,
      coneConstraints,
      varMap,
      objectiveOffset
    );

    // Call the solver
    const result = await solveConic(
      stuffed.P,
      stuffed.q,
      stuffed.A,
      stuffed.b,
      stuffed.coneDims,
      this._settings
    );

    // Handle error statuses
    if (result.status === 'infeasible') {
      throw new InfeasibleError('Problem is infeasible');
    }
    if (result.status === 'unbounded') {
      throw new UnboundedError('Problem is unbounded');
    }
    if (result.status === 'numerical_error') {
      throw new SolverError('Solver encountered numerical difficulties');
    }

    // Extract solution values for original variables
    let primal: Map<ExprId, Float64Array> | undefined;
    let value: number | undefined;

    if (result.status === 'optimal' && result.x) {
      primal = new Map();

      for (const varId of varIds) {
        const mapping = varMap.idToCol.get(varId);
        if (mapping) {
          const varValues = new Float64Array(mapping.size);
          for (let i = 0; i < mapping.size; i++) {
            varValues[i] = result.x[mapping.start + i] ?? 0;
          }
          primal.set(varId, varValues);
        }
      }

      // Compute objective value (add back offset, handle maximize)
      value = result.objVal ?? 0;
      value += objectiveOffset;

      // For maximization, negate back
      if (this._sense === 'maximize') {
        value = -value;
      }
    }

    return {
      status: result.status,
      value,
      primal,
      solveTime: result.solveTime,
      iterations: result.iterations,
    };
  }

  /**
   * Collect variable sizes from an expression.
   */
  private collectVariableSizes(expr: Expr, sizes: Map<ExprId, number>): void {
    if (expr.kind === 'variable') {
      sizes.set(expr.id, size(expr.shape));
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
        this.collectVariableSizes(expr.left, sizes);
        this.collectVariableSizes(expr.right, sizes);
        break;
      case 'neg':
      case 'sum':
      case 'reshape':
      case 'transpose':
      case 'trace':
      case 'diag':
      case 'norm1':
      case 'norm2':
      case 'normInf':
      case 'abs':
      case 'pos':
      case 'sumSquares':
        this.collectVariableSizes(expr.arg, sizes);
        break;
      case 'index':
        this.collectVariableSizes(expr.arg, sizes);
        break;
      case 'vstack':
      case 'hstack':
      case 'maximum':
      case 'minimum':
        for (const arg of expr.args) {
          this.collectVariableSizes(arg, sizes);
        }
        break;
      case 'quadForm':
        this.collectVariableSizes(expr.x, sizes);
        this.collectVariableSizes(expr.P, sizes);
        break;
      case 'quadOverLin':
        this.collectVariableSizes(expr.x, sizes);
        this.collectVariableSizes(expr.y, sizes);
        break;
    }
  }

  /**
   * Collect variable sizes from a constraint.
   */
  private collectVariableSizesFromConstraint(
    constraint: Constraint,
    sizes: Map<ExprId, number>
  ): void {
    switch (constraint.kind) {
      case 'zero':
      case 'nonneg':
        this.collectVariableSizes(constraint.expr, sizes);
        break;
      case 'soc':
        this.collectVariableSizes(constraint.t, sizes);
        this.collectVariableSizes(constraint.x, sizes);
        break;
    }
  }
}
