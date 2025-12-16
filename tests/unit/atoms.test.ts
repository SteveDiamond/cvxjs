import { describe, it, expect, beforeEach } from 'vitest';
import { variable, constant, exprShape, resetExprIds } from '../../src/index.js';
import {
  add,
  sub,
  neg,
  mul,
  div,
  matmul,
  sum,
  reshape,
  vstack,
  hstack,
  transpose,
  trace,
  diag,
  dot,
  norm1,
  norm2,
  normInf,
  norm,
  abs,
  pos,
  maximum,
  minimum,
  sumSquares,
  quadForm,
} from '../../src/atoms/index.js';
import { scalar, vector, matrix, shapeEquals } from '../../src/expr/shape.js';

describe('Affine atoms', () => {
  beforeEach(() => {
    resetExprIds();
  });

  describe('add', () => {
    it('adds two expressions', () => {
      const x = variable(5);
      const y = variable(5);
      const z = add(x, y);
      expect(z.data.kind).toBe('add');
      expect(shapeEquals(exprShape(z), vector(5))).toBe(true);
    });

    it('adds expression and number', () => {
      const x = variable(5);
      const z = add(x, 1);
      expect(z.data.kind).toBe('add');
    });

    it('throws on incompatible shapes', () => {
      const x = variable(5);
      const y = variable(6);
      expect(() => add(x, y)).toThrow();
    });
  });

  describe('sub', () => {
    it('subtracts two expressions', () => {
      const x = variable(5);
      const y = variable(5);
      const z = sub(x, y);
      expect(z.data.kind).toBe('add'); // sub is implemented as add(x, neg(y))
    });
  });

  describe('neg', () => {
    it('negates expression', () => {
      const x = variable(5);
      const y = neg(x);
      expect(y.data.kind).toBe('neg');
    });

    it('cancels double negation', () => {
      const x = variable(5);
      const y = neg(neg(x));
      expect(y.data).toBe(x.data); // Should return the same underlying Expr
    });
  });

  describe('mul', () => {
    it('multiplies expressions', () => {
      const x = variable(5);
      const c = constant(2);
      const z = mul(c, x);
      expect(z.data.kind).toBe('mul');
    });
  });

  describe('div', () => {
    it('divides expression by scalar', () => {
      const x = variable(5);
      const z = div(x, 2);
      expect(z.data.kind).toBe('div');
    });
  });

  describe('matmul', () => {
    it('performs matrix-vector multiplication', () => {
      const A = constant([
        [1, 2, 3],
        [4, 5, 6],
      ]); // 2x3
      const x = variable(3);
      const y = matmul(A, x);
      expect(y.data.kind).toBe('matmul');
      expect(shapeEquals(exprShape(y), vector(2))).toBe(true);
    });

    it('performs matrix-matrix multiplication', () => {
      const A = constant([
        [1, 2, 3],
        [4, 5, 6],
      ]); // 2x3
      const B = constant([
        [1, 2],
        [3, 4],
        [5, 6],
      ]); // 3x2
      const C = matmul(A, B);
      expect(shapeEquals(exprShape(C), matrix(2, 2))).toBe(true);
    });

    it('throws on incompatible dimensions', () => {
      const A = constant([
        [1, 2, 3],
        [4, 5, 6],
      ]); // 2x3
      const x = variable(4); // Wrong size
      expect(() => matmul(A, x)).toThrow();
    });
  });

  describe('sum', () => {
    it('sums all elements', () => {
      const x = variable(5);
      const s = sum(x);
      expect(s.data.kind).toBe('sum');
      expect(shapeEquals(exprShape(s), scalar())).toBe(true);
    });

    it('sums along axis', () => {
      const A = variable([3, 4]);
      const s0 = sum(A, 0); // Sum along rows -> 4 columns
      const s1 = sum(A, 1); // Sum along cols -> 3 rows
      expect(s0.data.kind).toBe('sum');
      expect(s1.data.kind).toBe('sum');
    });
  });

  describe('reshape', () => {
    it('reshapes vector to matrix', () => {
      const x = variable(12);
      const M = reshape(x, [3, 4]);
      expect(M.data.kind).toBe('reshape');
      expect(shapeEquals(exprShape(M), matrix(3, 4))).toBe(true);
    });

    it('throws on size mismatch', () => {
      const x = variable(10);
      expect(() => reshape(x, [3, 4])).toThrow();
    });
  });

  describe('vstack', () => {
    it('stacks vectors vertically', () => {
      const x = variable(3);
      const y = variable(3);
      const z = vstack(x, y);
      expect(z.data.kind).toBe('vstack');
    });

    it('returns single arg unchanged', () => {
      const x = variable(3);
      expect(vstack(x).data).toBe(x.data);
    });
  });

  describe('hstack', () => {
    it('stacks vectors horizontally', () => {
      const x = variable(3);
      const y = variable(3);
      const z = hstack(x, y);
      expect(z.data.kind).toBe('hstack');
    });
  });

  describe('transpose', () => {
    it('transposes matrix', () => {
      const A = variable([3, 4]);
      const At = transpose(A);
      expect(At.data.kind).toBe('transpose');
      expect(shapeEquals(exprShape(At), matrix(4, 3))).toBe(true);
    });

    it('cancels double transpose', () => {
      const A = variable([3, 4]);
      const Att = transpose(transpose(A));
      expect(Att.data).toBe(A.data);
    });
  });

  describe('trace', () => {
    it('computes trace of square matrix', () => {
      const A = variable([3, 3]);
      const t = trace(A);
      expect(t.data.kind).toBe('trace');
      expect(shapeEquals(exprShape(t), scalar())).toBe(true);
    });

    it('throws on non-square matrix', () => {
      const A = variable([3, 4]);
      expect(() => trace(A)).toThrow();
    });
  });

  describe('diag', () => {
    it('extracts diagonal from matrix', () => {
      const A = variable([3, 3]);
      const d = diag(A);
      expect(d.data.kind).toBe('diag');
      expect(shapeEquals(exprShape(d), vector(3))).toBe(true);
    });

    it('creates diagonal matrix from vector', () => {
      const v = variable(3);
      const D = diag(v);
      expect(D.data.kind).toBe('diag');
      expect(shapeEquals(exprShape(D), matrix(3, 3))).toBe(true);
    });
  });

  describe('dot', () => {
    it('computes dot product', () => {
      const x = variable(5);
      const y = variable(5);
      const d = dot(x, y);
      // dot is implemented as sum(mul(x, y))
      expect(d.data.kind).toBe('sum');
      expect(shapeEquals(exprShape(d), scalar())).toBe(true);
    });

    it('throws on non-vectors', () => {
      const A = variable([3, 4]);
      const B = variable([3, 4]);
      expect(() => dot(A, B)).toThrow();
    });

    it('throws on different lengths', () => {
      const x = variable(5);
      const y = variable(6);
      expect(() => dot(x, y)).toThrow();
    });
  });
});

describe('Nonlinear atoms', () => {
  beforeEach(() => {
    resetExprIds();
  });

  describe('norms', () => {
    it('norm1 returns scalar', () => {
      const x = variable(5);
      const n = norm1(x);
      expect(n.data.kind).toBe('norm1');
      expect(shapeEquals(exprShape(n), scalar())).toBe(true);
    });

    it('norm2 returns scalar', () => {
      const x = variable(5);
      const n = norm2(x);
      expect(n.data.kind).toBe('norm2');
      expect(shapeEquals(exprShape(n), scalar())).toBe(true);
    });

    it('normInf returns scalar', () => {
      const x = variable(5);
      const n = normInf(x);
      expect(n.data.kind).toBe('normInf');
      expect(shapeEquals(exprShape(n), scalar())).toBe(true);
    });

    it('norm() dispatches correctly', () => {
      const x = variable(5);
      expect(norm(x, 1).data.kind).toBe('norm1');
      expect(norm(x, 2).data.kind).toBe('norm2');
      expect(norm(x, Infinity).data.kind).toBe('normInf');
    });
  });

  describe('abs', () => {
    it('preserves shape', () => {
      const x = variable(5);
      const y = abs(x);
      expect(y.data.kind).toBe('abs');
      expect(shapeEquals(exprShape(y), vector(5))).toBe(true);
    });
  });

  describe('pos', () => {
    it('preserves shape', () => {
      const x = variable(5);
      const y = pos(x);
      expect(y.data.kind).toBe('pos');
      expect(shapeEquals(exprShape(y), vector(5))).toBe(true);
    });
  });

  describe('maximum/minimum', () => {
    it('maximum preserves shape', () => {
      const x = variable(5);
      const y = variable(5);
      const z = maximum(x, y);
      expect(z.data.kind).toBe('maximum');
      expect(shapeEquals(exprShape(z), vector(5))).toBe(true);
    });

    it('minimum preserves shape', () => {
      const x = variable(5);
      const y = variable(5);
      const z = minimum(x, y);
      expect(z.data.kind).toBe('minimum');
      expect(shapeEquals(exprShape(z), vector(5))).toBe(true);
    });

    it('maximum of single arg returns arg', () => {
      const x = variable(5);
      expect(maximum(x).data).toBe(x.data);
    });
  });

  describe('sumSquares', () => {
    it('returns scalar', () => {
      const x = variable(5);
      const s = sumSquares(x);
      expect(s.data.kind).toBe('sumSquares');
      expect(shapeEquals(exprShape(s), scalar())).toBe(true);
    });
  });

  describe('quadForm', () => {
    it('returns scalar', () => {
      const x = variable(3);
      const P = constant([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ]);
      const q = quadForm(x, P);
      expect(q.data.kind).toBe('quadForm');
      expect(shapeEquals(exprShape(q), scalar())).toBe(true);
    });

    it('throws on non-vector x', () => {
      const X = variable([3, 3]);
      const P = constant([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ]);
      expect(() => quadForm(X, P)).toThrow();
    });

    it('throws on incompatible P', () => {
      const x = variable(3);
      const P = constant([
        [1, 0],
        [0, 1],
      ]); // 2x2, should be 3x3
      expect(() => quadForm(x, P)).toThrow();
    });
  });
});
