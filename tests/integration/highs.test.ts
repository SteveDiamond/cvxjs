import { describe, it, expect, beforeEach } from 'vitest';
import {
  variable,
  constant,
  sum,
  norm2,
  mul,
  add,
  ge,
  le,
  eq,
  Problem,
  resetExprIds,
  loadHiGHS,
  DcpError,
} from '../../src/index.js';

describe('HiGHS Integration', () => {
  beforeEach(() => {
    resetExprIds();
  });

  describe('HiGHS loading', () => {
    it('loads HiGHS WASM successfully', async () => {
      const highs = await loadHiGHS();
      expect(highs).toBeDefined();
      expect(typeof highs.solve).toBe('function');
    });
  });

  describe('Integer Linear Programming', () => {
    it('solves simple integer LP', async () => {
      // minimize sum(x) subject to x >= 1, x integer
      // Optimal: x = [1, 1, 1], value = 3
      const x = variable(3, { integer: true });
      const solution = await Problem.minimize(sum(x))
        .subjectTo([ge(x, constant([1, 1, 1]))])
        .solve();

      expect(solution.status).toBe('optimal');
      expect(solution.value).toBeCloseTo(3, 5);

      const xVal = solution.valueOf(x);
      expect(xVal).toBeDefined();
      if (xVal) {
        expect(xVal[0]).toBeCloseTo(1, 5);
        expect(xVal[1]).toBeCloseTo(1, 5);
        expect(xVal[2]).toBeCloseTo(1, 5);
      }
    });

    it('solves integer LP with fractional constraints', async () => {
      // minimize x subject to x >= 1.5, x integer
      // Optimal: x = 2 (rounded up)
      const x = variable(1, { integer: true });
      const solution = await Problem.minimize(sum(x))
        .subjectTo([ge(x, constant([1.5]))])
        .solve();

      expect(solution.status).toBe('optimal');
      expect(solution.value).toBeCloseTo(2, 5);
    });

    it('solves integer LP maximization', async () => {
      // maximize sum(x) subject to x <= 2.5, x integer, x >= 0
      // Optimal: x = [2, 2, 2], value = 6
      const x = variable(3, { integer: true, nonneg: true });
      const solution = await Problem.maximize(sum(x))
        .subjectTo([le(x, constant([2.5, 2.5, 2.5]))])
        .solve();

      expect(solution.status).toBe('optimal');
      expect(solution.value).toBeCloseTo(6, 5);

      const xVal = solution.valueOf(x);
      expect(xVal).toBeDefined();
      if (xVal) {
        expect(xVal[0]).toBeCloseTo(2, 5);
        expect(xVal[1]).toBeCloseTo(2, 5);
        expect(xVal[2]).toBeCloseTo(2, 5);
      }
    });
  });

  describe('Binary Programming', () => {
    it('solves simple binary knapsack', async () => {
      // maximize 3*x[0] + 2*x[1] + 4*x[2]
      // subject to 2*x[0] + 1*x[1] + 3*x[2] <= 4
      //            x binary
      // Optimal: x = [0, 1, 1], value = 6
      const x = variable(3, { binary: true });
      // values = [3, 2, 4], weights = [2, 1, 3]
      const capacity = 4;

      // Element-wise multiply and sum for objective
      const objective = add(
        add(mul(constant(3), x.index(0)), mul(constant(2), x.index(1))),
        mul(constant(4), x.index(2))
      );

      // Element-wise multiply and sum for constraint
      const weightSum = add(
        add(mul(constant(2), x.index(0)), mul(constant(1), x.index(1))),
        mul(constant(3), x.index(2))
      );

      const solution = await Problem.maximize(objective)
        .subjectTo([le(weightSum, constant(capacity))])
        .solve();

      expect(solution.status).toBe('optimal');
      expect(solution.value).toBeCloseTo(6, 5);

      const xVal = solution.valueOf(x);
      expect(xVal).toBeDefined();
      if (xVal) {
        // Optimal: x = [0, 1, 1] or similar
        expect(xVal[0]).toBeCloseTo(0, 5);
        expect(xVal[1]).toBeCloseTo(1, 5);
        expect(xVal[2]).toBeCloseTo(1, 5);
      }
    });

    it('solves binary selection problem', async () => {
      // Select exactly 2 items from 4, minimize total cost
      // costs = [5, 3, 8, 2]
      // minimize sum(cost * x) subject to sum(x) = 2, x binary
      // Optimal: select items 2 and 4 (indices 1 and 3), value = 3 + 2 = 5
      const x = variable(4, { binary: true });
      // costs = [5, 3, 8, 2]

      // Objective: sum of selected costs
      const objective = add(
        add(mul(constant(5), x.index(0)), mul(constant(3), x.index(1))),
        add(mul(constant(8), x.index(2)), mul(constant(2), x.index(3)))
      );

      const solution = await Problem.minimize(objective)
        .subjectTo([eq(sum(x), constant(2))])
        .solve();

      expect(solution.status).toBe('optimal');
      expect(solution.value).toBeCloseTo(5, 5);

      const xVal = solution.valueOf(x);
      expect(xVal).toBeDefined();
      if (xVal) {
        // Should select x[1]=1 and x[3]=1
        expect(xVal[0]).toBeCloseTo(0, 5);
        expect(xVal[1]).toBeCloseTo(1, 5);
        expect(xVal[2]).toBeCloseTo(0, 5);
        expect(xVal[3]).toBeCloseTo(1, 5);
      }
    });
  });

  describe('Mixed Integer Programming', () => {
    it('solves MIP with continuous and integer variables', async () => {
      // minimize x + y
      // subject to x + y >= 3.5, x continuous nonneg, y integer nonneg
      // Optimal: x = 0.5, y = 3 (y rounds up to 3, x fills the gap)
      const x = variable(1, { nonneg: true }); // continuous
      const y = variable(1, { integer: true, nonneg: true }); // integer

      const objective = add(sum(x), sum(y));

      const solution = await Problem.minimize(objective)
        .subjectTo([ge(add(sum(x), sum(y)), constant(3.5))])
        .solve();

      expect(solution.status).toBe('optimal');
      expect(solution.value).toBeCloseTo(3.5, 5);

      const xVal = solution.valueOf(x);
      const yVal = solution.valueOf(y);
      expect(xVal).toBeDefined();
      expect(yVal).toBeDefined();

      // y should be integer, x can be fractional
      if (yVal) {
        expect(Math.round(yVal[0]!)).toBe(yVal[0]);
      }
    });
  });

  describe('Error handling', () => {
    it('throws error for integer + conic combination', async () => {
      // This should throw because norm2 is conic but x is integer
      const x = variable(3, { integer: true });

      await expect(
        Problem.minimize(norm2(x))
          .subjectTo([ge(x, constant([1, 1, 1]))])
          .solve()
      ).rejects.toThrow(DcpError);
    });

    it('throws error for binary + conic combination', async () => {
      // This should throw because norm2 is conic but x is binary
      const x = variable(3, { binary: true });

      await expect(
        Problem.minimize(norm2(x))
          .subjectTo([ge(x, constant([1, 1, 1]))])
          .solve()
      ).rejects.toThrow(DcpError);
    });
  });
});
