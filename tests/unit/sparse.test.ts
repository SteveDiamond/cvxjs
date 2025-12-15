import { describe, it, expect } from 'vitest';
import {
  cscEmpty,
  cscIdentity,
  cscFromTriplets,
  cscFromDense,
  cscNnz,
  cscGet,
  cscScale,
  cscAdd,
  cscSub,
  cscVstack,
  cscHstack,
  cscTranspose,
  cscMulVec,
  cscMulMat,
  cscDiag,
  cscToDense,
  cscClone,
  cscEquals,
} from '../../src/sparse/index.js';

describe('CSC Sparse Matrix', () => {
  describe('creation', () => {
    it('creates empty matrix', () => {
      const A = cscEmpty(3, 4);
      expect(A.nrows).toBe(3);
      expect(A.ncols).toBe(4);
      expect(cscNnz(A)).toBe(0);
    });

    it('creates identity matrix', () => {
      const I = cscIdentity(3);
      expect(I.nrows).toBe(3);
      expect(I.ncols).toBe(3);
      expect(cscNnz(I)).toBe(3);
      expect(cscGet(I, 0, 0)).toBe(1);
      expect(cscGet(I, 1, 1)).toBe(1);
      expect(cscGet(I, 2, 2)).toBe(1);
      expect(cscGet(I, 0, 1)).toBe(0);
    });

    it('creates from triplets', () => {
      const A = cscFromTriplets(
        3,
        3,
        [0, 1, 2], // rows
        [0, 1, 2], // cols
        [1, 2, 3] // values
      );
      expect(cscGet(A, 0, 0)).toBe(1);
      expect(cscGet(A, 1, 1)).toBe(2);
      expect(cscGet(A, 2, 2)).toBe(3);
    });

    it('sums duplicate triplets', () => {
      const A = cscFromTriplets(2, 2, [0, 0, 1], [0, 0, 1], [1, 2, 3]);
      expect(cscGet(A, 0, 0)).toBe(3); // 1 + 2
      expect(cscGet(A, 1, 1)).toBe(3);
    });

    it('creates from dense array', () => {
      // Column-major: [1, 2, 3, 4] for 2x2 matrix [[1,3],[2,4]]
      const data = new Float64Array([1, 2, 3, 4]);
      const A = cscFromDense(2, 2, data);
      expect(cscGet(A, 0, 0)).toBe(1);
      expect(cscGet(A, 1, 0)).toBe(2);
      expect(cscGet(A, 0, 1)).toBe(3);
      expect(cscGet(A, 1, 1)).toBe(4);
    });
  });

  describe('element access', () => {
    it('returns 0 for missing elements', () => {
      const A = cscFromTriplets(3, 3, [0], [0], [5]);
      expect(cscGet(A, 0, 0)).toBe(5);
      expect(cscGet(A, 1, 1)).toBe(0);
      expect(cscGet(A, 2, 2)).toBe(0);
    });
  });

  describe('scalar operations', () => {
    it('scales matrix', () => {
      const A = cscIdentity(3);
      const B = cscScale(A, 2);
      expect(cscGet(B, 0, 0)).toBe(2);
      expect(cscGet(B, 1, 1)).toBe(2);
    });

    it('scaling by 0 returns zero matrix', () => {
      const A = cscIdentity(3);
      const B = cscScale(A, 0);
      expect(cscNnz(B)).toBe(0);
    });
  });

  describe('matrix addition', () => {
    it('adds matrices', () => {
      const A = cscIdentity(2);
      const B = cscIdentity(2);
      const C = cscAdd(A, B);
      expect(cscGet(C, 0, 0)).toBe(2);
      expect(cscGet(C, 1, 1)).toBe(2);
    });

    it('subtracts matrices', () => {
      const A = cscScale(cscIdentity(2), 3);
      const B = cscIdentity(2);
      const C = cscSub(A, B);
      expect(cscGet(C, 0, 0)).toBe(2);
      expect(cscGet(C, 1, 1)).toBe(2);
    });

    it('throws on shape mismatch', () => {
      const A = cscIdentity(2);
      const B = cscIdentity(3);
      expect(() => cscAdd(A, B)).toThrow();
    });
  });

  describe('stacking', () => {
    it('vstacks matrices', () => {
      const A = cscFromTriplets(2, 2, [0, 1], [0, 1], [1, 2]);
      const B = cscFromTriplets(2, 2, [0, 1], [0, 1], [3, 4]);
      const C = cscVstack(A, B);

      expect(C.nrows).toBe(4);
      expect(C.ncols).toBe(2);
      expect(cscGet(C, 0, 0)).toBe(1);
      expect(cscGet(C, 1, 1)).toBe(2);
      expect(cscGet(C, 2, 0)).toBe(3);
      expect(cscGet(C, 3, 1)).toBe(4);
    });

    it('hstacks matrices', () => {
      const A = cscFromTriplets(2, 2, [0, 1], [0, 1], [1, 2]);
      const B = cscFromTriplets(2, 2, [0, 1], [0, 1], [3, 4]);
      const C = cscHstack(A, B);

      expect(C.nrows).toBe(2);
      expect(C.ncols).toBe(4);
      expect(cscGet(C, 0, 0)).toBe(1);
      expect(cscGet(C, 1, 1)).toBe(2);
      expect(cscGet(C, 0, 2)).toBe(3);
      expect(cscGet(C, 1, 3)).toBe(4);
    });
  });

  describe('transpose', () => {
    it('transposes matrix', () => {
      const A = cscFromTriplets(2, 3, [0, 1, 0], [0, 1, 2], [1, 2, 3]);
      const At = cscTranspose(A);

      expect(At.nrows).toBe(3);
      expect(At.ncols).toBe(2);
      expect(cscGet(At, 0, 0)).toBe(1);
      expect(cscGet(At, 1, 1)).toBe(2);
      expect(cscGet(At, 2, 0)).toBe(3);
    });
  });

  describe('multiplication', () => {
    it('multiplies matrix by vector', () => {
      const A = cscFromDense(2, 2, new Float64Array([1, 2, 3, 4]));
      const x = new Float64Array([1, 1]);
      const y = cscMulVec(A, x);

      // [1,3; 2,4] * [1;1] = [4; 6]
      expect(y[0]).toBe(4);
      expect(y[1]).toBe(6);
    });

    it('multiplies matrices', () => {
      const A = cscIdentity(2);
      const B = cscFromDense(2, 2, new Float64Array([1, 2, 3, 4]));
      const C = cscMulMat(A, B);

      expect(cscEquals(B, C)).toBe(true);
    });

    it('throws on dimension mismatch', () => {
      const A = cscIdentity(2);
      const x = new Float64Array([1, 2, 3]);
      expect(() => cscMulVec(A, x)).toThrow();
    });
  });

  describe('diagonal', () => {
    it('creates diagonal matrix from vector', () => {
      const v = new Float64Array([1, 2, 3]);
      const D = cscDiag(v);

      expect(D.nrows).toBe(3);
      expect(D.ncols).toBe(3);
      expect(cscGet(D, 0, 0)).toBe(1);
      expect(cscGet(D, 1, 1)).toBe(2);
      expect(cscGet(D, 2, 2)).toBe(3);
      expect(cscGet(D, 0, 1)).toBe(0);
    });
  });

  describe('conversion', () => {
    it('converts to dense', () => {
      const A = cscFromTriplets(2, 2, [0, 1, 0, 1], [0, 0, 1, 1], [1, 2, 3, 4]);
      const dense = cscToDense(A);

      expect(dense).toEqual(new Float64Array([1, 2, 3, 4]));
    });

    it('clones matrix', () => {
      const A = cscIdentity(3);
      const B = cscClone(A);

      expect(cscEquals(A, B)).toBe(true);
      expect(A.values).not.toBe(B.values); // Different arrays
    });
  });

  describe('equality', () => {
    it('compares equal matrices', () => {
      const A = cscIdentity(3);
      const B = cscIdentity(3);
      expect(cscEquals(A, B)).toBe(true);
    });

    it('compares unequal matrices', () => {
      const A = cscIdentity(3);
      const B = cscScale(cscIdentity(3), 2);
      expect(cscEquals(A, B)).toBe(false);
    });

    it('compares with tolerance', () => {
      const A = cscFromTriplets(1, 1, [0], [0], [1.0]);
      const B = cscFromTriplets(1, 1, [0], [0], [1.0 + 1e-12]);
      expect(cscEquals(A, B, 1e-10)).toBe(true);
    });
  });
});
