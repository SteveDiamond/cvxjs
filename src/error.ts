/**
 * Base error class for cvxjs.
 */
export class CvxError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'CvxError';
  }
}

/**
 * Error thrown when an expression violates DCP rules.
 */
export class DcpError extends CvxError {
  constructor(message: string) {
    super(message);
    this.name = 'DcpError';
  }
}

/**
 * Error thrown when shapes are incompatible.
 */
export class ShapeError extends CvxError {
  readonly expected: string;
  readonly got: string;

  constructor(message: string, expected: string, got: string) {
    super(`${message}: expected ${expected}, got ${got}`);
    this.name = 'ShapeError';
    this.expected = expected;
    this.got = got;
  }
}

/**
 * Error thrown when solver encounters an issue.
 */
export class SolverError extends CvxError {
  constructor(message: string) {
    super(message);
    this.name = 'SolverError';
  }
}

/**
 * Error thrown when problem is infeasible.
 */
export class InfeasibleError extends SolverError {
  constructor(message = 'Problem is infeasible') {
    super(message);
    this.name = 'InfeasibleError';
  }
}

/**
 * Error thrown when problem is unbounded.
 */
export class UnboundedError extends SolverError {
  constructor(message = 'Problem is unbounded') {
    super(message);
    this.name = 'UnboundedError';
  }
}
