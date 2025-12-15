import { describe, it, expect, beforeEach } from 'vitest';
import { variable, eq, le, ge, Problem, resetExprIds, DcpError } from '../../src/index.js';
import { sum, norm2, sumSquares, minimum, neg } from '../../src/atoms/index.js';

describe('Problem', () => {
  beforeEach(() => {
    resetExprIds();
  });

  describe('creation', () => {
    it('creates minimization problem', () => {
      const x = variable(5);
      const p = Problem.minimize(sum(x));
      expect(p.sense).toBe('minimize');
    });

    it('creates maximization problem', () => {
      const x = variable(5);
      const p = Problem.maximize(neg(sumSquares(x))); // concave objective
      expect(p.sense).toBe('maximize');
    });

    it('throws on non-scalar objective', () => {
      const x = variable(5);
      expect(() => Problem.minimize(x)).toThrow(DcpError);
    });
  });

  describe('builder pattern', () => {
    it('adds constraints', () => {
      const x = variable(5);
      const p = Problem.minimize(sum(x))
        .subjectTo([ge(x, 0)])
        .subjectTo([le(x, 10)]);

      expect(p.constraints.length).toBe(2);
    });

    it('sets solver settings', () => {
      const x = variable(5);
      const p = Problem.minimize(sum(x)).settings({ verbose: true, maxIter: 100 });

      // Settings are stored internally (no public accessor yet)
      expect(p).toBeDefined();
    });
  });

  describe('DCP validation', () => {
    it('accepts convex minimization', () => {
      const x = variable(5);
      const p = Problem.minimize(sumSquares(x)).subjectTo([ge(x, 1)]);

      expect(p.isDcp()).toBe(true);
    });

    it('accepts affine minimization', () => {
      const x = variable(5);
      const p = Problem.minimize(sum(x)).subjectTo([ge(x, 0)]);

      expect(p.isDcp()).toBe(true);
    });

    it('rejects concave minimization', () => {
      const x = variable(5);
      const y = variable(5);
      // minimum is concave, can't minimize concave
      const p = Problem.minimize(minimum(sum(x), sum(y)));

      expect(p.isDcp()).toBe(false);
    });

    it('accepts concave maximization', () => {
      const x = variable(5);
      // -||x||^2 is concave
      const p = Problem.maximize(neg(sumSquares(x))).subjectTo([ge(x, 0)]);

      expect(p.isDcp()).toBe(true);
    });

    it('rejects convex maximization', () => {
      const x = variable(5);
      // ||x||^2 is convex, can't maximize convex
      const p = Problem.maximize(sumSquares(x));

      expect(p.isDcp()).toBe(false);
    });

    it('validates constraints', () => {
      const x = variable(5);
      // Valid problem
      const p1 = Problem.minimize(sum(x)).subjectTo([eq(sum(x), 5), ge(x, 0)]);
      expect(p1.isDcp()).toBe(true);

      // Invalid constraint (equality with convex)
      const p2 = Problem.minimize(sum(x)).subjectTo([eq(norm2(x), 1)]);
      expect(p2.isDcp()).toBe(false);
    });
  });

  describe('variables', () => {
    it('collects all variables', () => {
      const x = variable(5);
      const y = variable(3);
      const p = Problem.minimize(sum(x)).subjectTo([ge(y, 0)]);

      expect(p.variables.size).toBe(2);
    });
  });

  describe('validateDcp', () => {
    it('throws on invalid objective', () => {
      const x = variable(5);
      const y = variable(5);
      const p = Problem.minimize(minimum(sum(x), sum(y)));

      expect(() => p.validateDcp()).toThrow(DcpError);
    });

    it('throws on invalid constraint', () => {
      const x = variable(5);
      const p = Problem.minimize(sum(x)).subjectTo([eq(norm2(x), 1)]);

      expect(() => p.validateDcp()).toThrow(DcpError);
    });

    it('does not throw on valid problem', () => {
      const x = variable(5);
      const p = Problem.minimize(sumSquares(x)).subjectTo([ge(x, 1)]);

      expect(() => p.validateDcp()).not.toThrow();
    });
  });
});
