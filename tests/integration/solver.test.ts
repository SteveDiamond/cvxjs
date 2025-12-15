import { describe, it, expect, beforeEach } from 'vitest';
import {
  variable,
  constant,
  sum,
  norm2,
  norm1,
  neg,
  mul,
  ge,
  le,
  eq,
  Problem,
  resetExprIds,
  testWasm,
  clarabelVersion,
} from '../../src/index.js';

describe('Solver Integration', () => {
  beforeEach(() => {
    resetExprIds();
  });

  describe('WASM loading', () => {
    it('loads WASM successfully', async () => {
      const result = await testWasm();
      expect(result).toBe('Clarabel WASM is working!');
    });

    it('gets Clarabel version', async () => {
      const version = await clarabelVersion();
      expect(version).toContain('clarabel-wasm');
    });
  });

  describe('Linear Programming', () => {
    it('solves unconstrained minimization', async () => {
      // minimize sum(x) where x is 3-vector
      // Unbounded in general, but with bounds it works
      const x = variable(3);
      const solution = await Problem.minimize(sum(x))
        .subjectTo([ge(x, constant([0, 0, 0]))])
        .solve();

      expect(solution.status).toBe('optimal');
      expect(solution.value).toBeCloseTo(0, 5);
    });

    it('solves simple LP with lower bounds', async () => {
      // minimize sum(x) subject to x >= 1
      // Optimal: x = [1, 1, 1], value = 3
      const x = variable(3);
      const solution = await Problem.minimize(sum(x))
        .subjectTo([ge(x, constant([1, 1, 1]))])
        .solve();

      expect(solution.status).toBe('optimal');
      expect(solution.value).toBeCloseTo(3, 5);

      const primal = solution.primal;
      expect(primal).toBeDefined();
      if (primal) {
        const xVal = primal.values().next().value;
        expect(xVal).toBeDefined();
        if (xVal) {
          expect(xVal[0]).toBeCloseTo(1, 5);
          expect(xVal[1]).toBeCloseTo(1, 5);
          expect(xVal[2]).toBeCloseTo(1, 5);
        }
      }
    });

    it('solves LP with upper bounds', async () => {
      // maximize sum(x) subject to x <= 2
      // Optimal: x = [2, 2, 2], value = 6
      const x = variable(3);
      const solution = await Problem.maximize(sum(x))
        .subjectTo([le(x, constant([2, 2, 2])), ge(x, constant([0, 0, 0]))])
        .solve();

      expect(solution.status).toBe('optimal');
      expect(solution.value).toBeCloseTo(6, 5);
    });

    it('solves LP with equality constraints', async () => {
      // minimize x[0] - x[1] subject to x[0] + x[1] = 2, x >= 0
      // Optimal: x = [0, 2], value = -2
      const x = variable(2);

      const obj = sum(mul(constant([1, -1]), x));
      const sumConstraint = eq(sum(x), constant(2));

      const solution = await Problem.minimize(obj)
        .subjectTo([sumConstraint, ge(x, constant([0, 0]))])
        .solve();

      expect(solution.status).toBe('optimal');
      expect(solution.value).toBeCloseTo(-2, 5);
    });
  });

  describe('Second-Order Cone Programming', () => {
    it('solves simple norm2 minimization', async () => {
      // minimize ||x||_2 subject to sum(x) = 3
      // Optimal: x = [1, 1, 1], ||x||_2 = sqrt(3)
      const x = variable(3);
      const solution = await Problem.minimize(norm2(x))
        .subjectTo([eq(sum(x), constant(3))])
        .solve();

      expect(solution.status).toBe('optimal');
      expect(solution.value).toBeCloseTo(Math.sqrt(3), 4);
    });

    it('solves norm2 with bounds', async () => {
      // minimize ||x||_2 subject to x >= 1
      // Optimal: x = [1, 1, 1], ||x||_2 = sqrt(3)
      const x = variable(3);
      const solution = await Problem.minimize(norm2(x))
        .subjectTo([ge(x, constant([1, 1, 1]))])
        .solve();

      expect(solution.status).toBe('optimal');
      expect(solution.value).toBeCloseTo(Math.sqrt(3), 4);
    });
  });

  describe('Norm1 Minimization', () => {
    it('solves norm1 minimization', async () => {
      // minimize ||x||_1 subject to sum(x) = 3
      // Many optima, but ||x||_1 >= 3 (achieved when all positive)
      const x = variable(3);
      const solution = await Problem.minimize(norm1(x))
        .subjectTo([eq(sum(x), constant(3)), ge(x, constant([0, 0, 0]))])
        .solve();

      expect(solution.status).toBe('optimal');
      expect(solution.value).toBeCloseTo(3, 4);
    });
  });

  describe('Error cases', () => {
    it('throws DcpError for non-convex objective', async () => {
      const x = variable(3);
      // -||x||_2 is concave, can't minimize
      const problem = Problem.minimize(neg(norm2(x)));

      // Solving should fail at DCP check
      await expect(problem.solve()).rejects.toThrow();
    });
  });
});
