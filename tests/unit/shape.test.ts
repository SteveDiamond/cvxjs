import { describe, it, expect } from 'vitest';
import {
  scalar,
  vector,
  matrix,
  size,
  rows,
  cols,
  isScalar,
  isVector,
  isMatrix,
  shapeEquals,
  shapeToString,
  broadcastShape,
  normalizeShape,
} from '../../src/expr/shape.js';

describe('Shape', () => {
  describe('creation', () => {
    it('creates scalar shape', () => {
      const s = scalar();
      expect(s.dims).toEqual([]);
    });

    it('creates vector shape', () => {
      const v = vector(5);
      expect(v.dims).toEqual([5]);
    });

    it('creates matrix shape', () => {
      const m = matrix(3, 4);
      expect(m.dims).toEqual([3, 4]);
    });

    it('throws for invalid dimensions', () => {
      expect(() => vector(0)).toThrow();
      expect(() => vector(-1)).toThrow();
      expect(() => matrix(0, 5)).toThrow();
      expect(() => matrix(5, -1)).toThrow();
    });
  });

  describe('size', () => {
    it('returns 1 for scalar', () => {
      expect(size(scalar())).toBe(1);
    });

    it('returns length for vector', () => {
      expect(size(vector(5))).toBe(5);
    });

    it('returns rows * cols for matrix', () => {
      expect(size(matrix(3, 4))).toBe(12);
    });
  });

  describe('rows and cols', () => {
    it('returns 1 for scalar', () => {
      expect(rows(scalar())).toBe(1);
      expect(cols(scalar())).toBe(1);
    });

    it('returns n and 1 for vector', () => {
      expect(rows(vector(5))).toBe(5);
      expect(cols(vector(5))).toBe(1);
    });

    it('returns correct values for matrix', () => {
      expect(rows(matrix(3, 4))).toBe(3);
      expect(cols(matrix(3, 4))).toBe(4);
    });
  });

  describe('type checks', () => {
    it('isScalar', () => {
      expect(isScalar(scalar())).toBe(true);
      expect(isScalar(vector(5))).toBe(false);
      expect(isScalar(matrix(3, 4))).toBe(false);
    });

    it('isVector', () => {
      expect(isVector(scalar())).toBe(false);
      expect(isVector(vector(5))).toBe(true);
      expect(isVector(matrix(3, 4))).toBe(false);
    });

    it('isMatrix', () => {
      expect(isMatrix(scalar())).toBe(false);
      expect(isMatrix(vector(5))).toBe(false);
      expect(isMatrix(matrix(3, 4))).toBe(true);
    });
  });

  describe('shapeEquals', () => {
    it('returns true for equal shapes', () => {
      expect(shapeEquals(scalar(), scalar())).toBe(true);
      expect(shapeEquals(vector(5), vector(5))).toBe(true);
      expect(shapeEquals(matrix(3, 4), matrix(3, 4))).toBe(true);
    });

    it('returns false for different shapes', () => {
      expect(shapeEquals(scalar(), vector(1))).toBe(false);
      expect(shapeEquals(vector(5), vector(6))).toBe(false);
      expect(shapeEquals(matrix(3, 4), matrix(4, 3))).toBe(false);
    });
  });

  describe('shapeToString', () => {
    it('formats shapes correctly', () => {
      expect(shapeToString(scalar())).toBe('scalar');
      expect(shapeToString(vector(5))).toBe('(5)');
      expect(shapeToString(matrix(3, 4))).toBe('(3, 4)');
    });
  });

  describe('broadcastShape', () => {
    it('broadcasts scalar to anything', () => {
      expect(broadcastShape(scalar(), vector(5))).toEqual(vector(5));
      expect(broadcastShape(vector(5), scalar())).toEqual(vector(5));
      expect(broadcastShape(scalar(), matrix(3, 4))).toEqual(matrix(3, 4));
    });

    it('broadcasts compatible shapes', () => {
      expect(broadcastShape(vector(5), vector(5))).toEqual(vector(5));
      expect(broadcastShape(matrix(3, 4), matrix(3, 4))).toEqual(matrix(3, 4));
    });

    it('returns null for incompatible shapes', () => {
      expect(broadcastShape(vector(5), vector(6))).toBe(null);
      expect(broadcastShape(matrix(3, 4), matrix(4, 3))).toBe(null);
    });
  });

  describe('normalizeShape', () => {
    it('normalizes number to vector', () => {
      expect(normalizeShape(5)).toEqual(vector(5));
    });

    it('normalizes [n] to vector', () => {
      expect(normalizeShape([5])).toEqual(vector(5));
    });

    it('normalizes [m, n] to matrix', () => {
      expect(normalizeShape([3, 4])).toEqual(matrix(3, 4));
    });

    it('passes through Shape objects', () => {
      const s = matrix(3, 4);
      expect(normalizeShape(s)).toBe(s);
    });
  });
});
