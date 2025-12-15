import { describe, it, expect, beforeEach } from 'vitest';
import {
  variable,
  constant,
  Curvature,
  curvature,
  isConvex,
  isConcave,
  isAffine,
  Sign,
  sign,
  isNonnegative,
  isNonpositive,
  resetExprIds,
} from '../../src/index.js';
import {
  add,
  neg,
  mul,
  sum,
  matmul,
  norm1,
  norm2,
  normInf,
  abs,
  pos,
  maximum,
  minimum,
  sumSquares,
  quadForm,
} from '../../src/atoms/index.js';

describe('Curvature', () => {
  beforeEach(() => {
    resetExprIds();
  });

  describe('leaf nodes', () => {
    it('variable is affine', () => {
      const x = variable(5);
      expect(curvature(x)).toBe(Curvature.Affine);
    });

    it('constant is constant', () => {
      const c = constant(5);
      expect(curvature(c)).toBe(Curvature.Constant);
    });
  });

  describe('affine operations', () => {
    it('add preserves curvature', () => {
      const x = variable(5);
      const y = variable(5);
      expect(curvature(add(x, y))).toBe(Curvature.Affine);

      const c = constant([1, 2, 3, 4, 5]);
      expect(curvature(add(x, c))).toBe(Curvature.Affine);
    });

    it('neg negates curvature', () => {
      const x = variable(5);
      expect(curvature(neg(x))).toBe(Curvature.Affine);
    });

    it('mul by constant preserves curvature', () => {
      const x = variable(5);
      const c = constant(2);
      expect(curvature(mul(c, x))).toBe(Curvature.Affine);
    });

    it('sum preserves curvature', () => {
      const x = variable(5);
      expect(curvature(sum(x))).toBe(Curvature.Affine);
    });

    it('matmul with constant preserves curvature', () => {
      const A = constant([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const x = variable(3);
      expect(curvature(matmul(A, x))).toBe(Curvature.Affine);
    });
  });

  describe('nonlinear convex atoms', () => {
    it('norm1 of affine is convex', () => {
      const x = variable(5);
      expect(curvature(norm1(x))).toBe(Curvature.Convex);
    });

    it('norm2 of affine is convex', () => {
      const x = variable(5);
      expect(curvature(norm2(x))).toBe(Curvature.Convex);
    });

    it('normInf of affine is convex', () => {
      const x = variable(5);
      expect(curvature(normInf(x))).toBe(Curvature.Convex);
    });

    it('abs of affine is convex', () => {
      const x = variable(5);
      expect(curvature(abs(x))).toBe(Curvature.Convex);
    });

    it('pos of affine is convex', () => {
      const x = variable(5);
      expect(curvature(pos(x))).toBe(Curvature.Convex);
    });

    it('sumSquares of affine is convex', () => {
      const x = variable(5);
      expect(curvature(sumSquares(x))).toBe(Curvature.Convex);
    });

    it('quadForm with constant P is convex', () => {
      const x = variable(3);
      const P = constant([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ]);
      expect(curvature(quadForm(x, P))).toBe(Curvature.Convex);
    });

    it('maximum of convex is convex', () => {
      const x = variable(5);
      const y = variable(5);
      expect(curvature(maximum(norm1(x), norm1(y)))).toBe(Curvature.Convex);
    });
  });

  describe('nonlinear concave atoms', () => {
    it('minimum of affine is concave', () => {
      const x = variable(5);
      const y = variable(5);
      expect(curvature(minimum(x, y))).toBe(Curvature.Concave);
    });
  });

  describe('DCP violations', () => {
    it('norm of convex is unknown', () => {
      const x = variable(5);
      expect(curvature(norm2(norm2(x)))).toBe(Curvature.Unknown);
    });

    it('variable times variable is unknown', () => {
      const x = variable(5);
      const y = variable(5);
      expect(curvature(mul(x, y))).toBe(Curvature.Unknown);
    });

    it('convex + concave is unknown', () => {
      const x = variable(5);
      const y = variable(5);
      const _convexPart = norm2(x);
      const _concavePart = minimum(y, constant([0, 0, 0, 0, 0]));
      // Note: This would actually be unknown due to shape mismatch
      // Let's test with scalars
      const a = variable(1);
      const b = variable(1);
      // sum is scalar
      const cv = sumSquares(a); // convex scalar
      const cc = minimum(b, constant(0)); // concave scalar
      expect(curvature(add(cv, cc))).toBe(Curvature.Unknown);
    });
  });

  describe('curvature helpers', () => {
    it('isConvex', () => {
      expect(isConvex(Curvature.Constant)).toBe(true);
      expect(isConvex(Curvature.Affine)).toBe(true);
      expect(isConvex(Curvature.Convex)).toBe(true);
      expect(isConvex(Curvature.Concave)).toBe(false);
      expect(isConvex(Curvature.Unknown)).toBe(false);
    });

    it('isConcave', () => {
      expect(isConcave(Curvature.Constant)).toBe(true);
      expect(isConcave(Curvature.Affine)).toBe(true);
      expect(isConcave(Curvature.Convex)).toBe(false);
      expect(isConcave(Curvature.Concave)).toBe(true);
      expect(isConcave(Curvature.Unknown)).toBe(false);
    });

    it('isAffine', () => {
      expect(isAffine(Curvature.Constant)).toBe(true);
      expect(isAffine(Curvature.Affine)).toBe(true);
      expect(isAffine(Curvature.Convex)).toBe(false);
      expect(isAffine(Curvature.Concave)).toBe(false);
      expect(isAffine(Curvature.Unknown)).toBe(false);
    });
  });
});

describe('Sign', () => {
  beforeEach(() => {
    resetExprIds();
  });

  describe('leaf nodes', () => {
    it('unconstrained variable has unknown sign', () => {
      const x = variable(5);
      expect(sign(x)).toBe(Sign.Unknown);
    });

    it('nonneg variable has nonnegative sign', () => {
      const x = variable(5, { nonneg: true });
      expect(sign(x)).toBe(Sign.Nonnegative);
    });

    it('nonpos variable has nonpositive sign', () => {
      const x = variable(5, { nonpos: true });
      expect(sign(x)).toBe(Sign.Nonpositive);
    });

    it('positive constant has nonnegative sign', () => {
      expect(sign(constant(5))).toBe(Sign.Nonnegative);
    });

    it('negative constant has nonpositive sign', () => {
      expect(sign(constant(-5))).toBe(Sign.Nonpositive);
    });

    it('zero constant has zero sign', () => {
      expect(sign(constant(0))).toBe(Sign.Zero);
    });
  });

  describe('operations', () => {
    it('neg negates sign', () => {
      const x = variable(5, { nonneg: true });
      expect(sign(neg(x))).toBe(Sign.Nonpositive);
    });

    it('abs is always nonnegative', () => {
      const x = variable(5);
      expect(sign(abs(x))).toBe(Sign.Nonnegative);
    });

    it('norm is always nonnegative', () => {
      const x = variable(5);
      expect(sign(norm1(x))).toBe(Sign.Nonnegative);
      expect(sign(norm2(x))).toBe(Sign.Nonnegative);
      expect(sign(normInf(x))).toBe(Sign.Nonnegative);
    });

    it('sumSquares is always nonnegative', () => {
      const x = variable(5);
      expect(sign(sumSquares(x))).toBe(Sign.Nonnegative);
    });

    it('pos is always nonnegative', () => {
      const x = variable(5);
      expect(sign(pos(x))).toBe(Sign.Nonnegative);
    });
  });

  describe('sign helpers', () => {
    it('isNonnegative', () => {
      expect(isNonnegative(Sign.Nonnegative)).toBe(true);
      expect(isNonnegative(Sign.Zero)).toBe(true);
      expect(isNonnegative(Sign.Nonpositive)).toBe(false);
      expect(isNonnegative(Sign.Unknown)).toBe(false);
    });

    it('isNonpositive', () => {
      expect(isNonpositive(Sign.Nonpositive)).toBe(true);
      expect(isNonpositive(Sign.Zero)).toBe(true);
      expect(isNonpositive(Sign.Nonnegative)).toBe(false);
      expect(isNonpositive(Sign.Unknown)).toBe(false);
    });
  });
});
