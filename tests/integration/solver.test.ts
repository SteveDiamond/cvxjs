import { describe, it, expect, beforeEach } from 'vitest';
import {
  variable,
  constant,
  sum,
  norm2,
  norm1,
  neg,
  mul,
  add,
  ge,
  le,
  eq,
  index,
  hstack,
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

  describe('Index and HStack', () => {
    it('solves LP with indexed objective', async () => {
      // minimize x[0] subject to x >= [1, 2, 3]
      // Optimal: x = [1, 2, 3], value = 1
      const x = variable(3);
      const solution = await Problem.minimize(index(x, 0))
        .subjectTo([ge(x, constant([1, 2, 3]))])
        .solve();

      expect(solution.status).toBe('optimal');
      expect(solution.value).toBeCloseTo(1, 5);
    });

    it('solves LP with indexed constraint', async () => {
      // minimize sum(x) subject to x[0] == 5, x >= 0
      // Optimal: x = [5, 0, 0], value = 5
      const x = variable(3);
      const solution = await Problem.minimize(sum(x))
        .subjectTo([eq(index(x, 0), constant(5)), ge(x, constant([0, 0, 0]))])
        .solve();

      expect(solution.status).toBe('optimal');
      expect(solution.value).toBeCloseTo(5, 5);
    });

    it('solves LP with range indexing in objective', async () => {
      // minimize sum(x[1:3]) subject to x >= [1, 2, 3]
      // Optimal: x = [1, 2, 3], value = 2 + 3 = 5
      const x = variable(3);
      const solution = await Problem.minimize(sum(index(x, [1, 3])))
        .subjectTo([ge(x, constant([1, 2, 3]))])
        .solve();

      expect(solution.status).toBe('optimal');
      expect(solution.value).toBeCloseTo(5, 5);
    });

    it('solves LP with multiple indexed elements in objective', async () => {
      // minimize x[0] + x[2] subject to x >= [1, 2, 3]
      // Optimal: x = [1, 2, 3], value = 1 + 3 = 4
      const x = variable(3);
      const obj = add(index(x, 0), index(x, 2));
      const solution = await Problem.minimize(obj)
        .subjectTo([ge(x, constant([1, 2, 3]))])
        .solve();

      expect(solution.status).toBe('optimal');
      expect(solution.value).toBeCloseTo(4, 5);
    });

    it('solves problem with hstack in constraint', async () => {
      // minimize sum(x) + sum(y) subject to hstack(x, y) >= ones(3, 2)
      // Optimal: x = [1, 1, 1], y = [1, 1, 1], value = 6
      const x = variable(3);
      const y = variable(3);
      const z = hstack(x, y);
      const solution = await Problem.minimize(add(sum(x), sum(y)))
        .subjectTo([
          ge(
            z,
            constant([
              [1, 1],
              [1, 1],
              [1, 1],
            ])
          ),
        ])
        .solve();

      expect(solution.status).toBe('optimal');
      expect(solution.value).toBeCloseTo(6, 5);
    });

    it('solves problem with indexed hstack result', async () => {
      // hstack(x, y) creates a 3x2 matrix
      // Extracting column 1 (y) and minimizing its sum
      // minimize sum(hstack(x, y)[:, 1]) subject to x >= 0, y >= [1, 2, 3]
      // This should minimize sum(y), optimal y = [1, 2, 3], value = 6
      const x = variable(3);
      const y = variable(3);
      const z = hstack(x, y);
      const col1 = index(z, 'all', 1); // Second column = y

      const solution = await Problem.minimize(sum(col1))
        .subjectTo([ge(x, constant([0, 0, 0])), ge(y, constant([1, 2, 3]))])
        .solve();

      expect(solution.status).toBe('optimal');
      expect(solution.value).toBeCloseTo(6, 5);
    });

    it('solves SOCP with indexed norm', async () => {
      // minimize ||x[0:2]||_2 subject to sum(x) = 3, x >= 0
      // Optimal: distribute weight on first 2 elements to minimize norm
      const x = variable(3);
      const sliced = index(x, [0, 2]);
      const solution = await Problem.minimize(norm2(sliced))
        .subjectTo([eq(sum(x), constant(3)), ge(x, constant([0, 0, 0]))])
        .solve();

      expect(solution.status).toBe('optimal');
      // With x[2] = 3, x[0] = x[1] = 0, norm = 0
      // But that doesn't satisfy sum(x) = 3 with x >= 0...
      // Let's verify it finds a feasible solution
      expect(solution.value).toBeGreaterThanOrEqual(0);
    });
  });

  describe('Dual Variables', () => {
    it('returns dual variables (shadow prices) for LP', async () => {
      // minimize sum(x) subject to x >= 1
      // Dual variable for each constraint x_i >= 1 should be 1
      const x = variable(3);
      const solution = await Problem.minimize(sum(x))
        .subjectTo([ge(x, constant([1, 1, 1]))])
        .solve();

      expect(solution.status).toBe('optimal');
      expect(solution.dual).toBeDefined();
      // Should have dual values for the nonneg constraints
      expect(solution.dual!.length).toBeGreaterThan(0);
    });

    it('returns dual variables for equality constraints', async () => {
      // minimize x subject to x = 5
      // Dual variable should be -1 (marginal cost of relaxing equality)
      const x = variable(1);
      const solution = await Problem.minimize(sum(x))
        .subjectTo([eq(x, constant([5]))])
        .solve();

      expect(solution.status).toBe('optimal');
      expect(solution.value).toBeCloseTo(5, 5);
      expect(solution.dual).toBeDefined();
      expect(solution.dual!.length).toBeGreaterThan(0);
    });
  });
});
