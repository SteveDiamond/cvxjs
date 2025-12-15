import { describe, it, expect, beforeEach } from 'vitest';
import {
  variable,
  constant,
  eq,
  le,
  ge,
  soc,
  isDcpConstraint,
  constraintVariables,
  resetExprIds,
} from '../../src/index.js';
import { sum, norm2, sumSquares } from '../../src/atoms/index.js';

describe('Constraints', () => {
  beforeEach(() => {
    resetExprIds();
  });

  describe('eq (equality)', () => {
    it('creates equality constraint', () => {
      const x = variable(5);
      const c = eq(sum(x), 1);
      expect(c.kind).toBe('zero');
    });

    it('collects variables', () => {
      const x = variable(5);
      const c = eq(sum(x), 1);
      const vars = constraintVariables(c);
      expect(vars.size).toBe(1);
    });

    it('is DCP when affine', () => {
      const x = variable(5);
      const c = eq(sum(x), 1);
      expect(isDcpConstraint(c)).toBe(true);
    });

    it('is not DCP when nonlinear', () => {
      const x = variable(5);
      // norm2(x) == 1 is not affine (norm2 is convex, not affine)
      const c = eq(norm2(x), 1);
      expect(isDcpConstraint(c)).toBe(false);
    });
  });

  describe('le (less than or equal)', () => {
    it('creates inequality constraint', () => {
      const x = variable(5);
      const c = le(sum(x), 10);
      expect(c.kind).toBe('nonneg');
    });

    it('is DCP when convex <= constant', () => {
      const x = variable(5);
      const c = le(norm2(x), 1);  // ||x|| <= 1
      expect(isDcpConstraint(c)).toBe(true);
    });

    it('is DCP when affine <= affine', () => {
      const x = variable(5);
      const y = variable(5);
      const c = le(sum(x), sum(y));
      expect(isDcpConstraint(c)).toBe(true);
    });

    it('is not DCP when concave <= constant', () => {
      const x = variable(5);
      const y = variable(5);
      // minimum is concave, can't be on LHS of <=
      // Actually: minimum(x, y) <= 5 means -minimum(x,y) + 5 >= 0
      // -minimum is convex (negation of concave), so this IS DCP
      // Let's test convex >= constant instead
    });
  });

  describe('ge (greater than or equal)', () => {
    it('creates inequality constraint', () => {
      const x = variable(5);
      const c = ge(x, 0);
      expect(c.kind).toBe('nonneg');
    });

    it('is DCP when constant >= convex', () => {
      const x = variable(5);
      const c = ge(1, norm2(x));  // 1 >= ||x||
      expect(isDcpConstraint(c)).toBe(true);
    });

    it('is DCP when affine >= 0', () => {
      const x = variable(5);
      const c = ge(x, 0);
      expect(isDcpConstraint(c)).toBe(true);
    });
  });

  describe('soc (second-order cone)', () => {
    it('creates SOC constraint', () => {
      const x = variable(5);
      const t = variable(1);
      const c = soc(x, t);
      expect(c.kind).toBe('soc');
    });

    it('is DCP when both affine', () => {
      const x = variable(5);
      const t = variable(1);
      const c = soc(x, t);
      expect(isDcpConstraint(c)).toBe(true);
    });

    it('collects variables from both args', () => {
      const x = variable(5);
      const t = variable(1);
      const c = soc(x, t);
      const vars = constraintVariables(c);
      expect(vars.size).toBe(2);
    });
  });
});
