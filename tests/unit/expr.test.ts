import { describe, it, expect, beforeEach } from 'vitest';
import {
  variable,
  scalarVar,
  vectorVar,
  matrixVar,
  constant,
  zeros,
  ones,
  eye,
  exprShape,
  exprVariables,
  isConstantExpr,
  isVariable,
  isConstant,
  resetExprIds,
} from '../../src/index.js';
import { add, neg, sum, matmul, transpose } from '../../src/atoms/index.js';
import { scalar, vector, matrix, shapeEquals } from '../../src/expr/shape.js';

describe('Variable', () => {
  beforeEach(() => {
    resetExprIds();
  });

  it('creates scalar variable', () => {
    const x = scalarVar();
    expect(isVariable(x)).toBe(true);
    expect(shapeEquals(exprShape(x), scalar())).toBe(true);
  });

  it('creates vector variable', () => {
    const x = vectorVar(5);
    expect(isVariable(x)).toBe(true);
    expect(shapeEquals(exprShape(x), vector(5))).toBe(true);
  });

  it('creates matrix variable', () => {
    const X = matrixVar(3, 4);
    expect(isVariable(X)).toBe(true);
    expect(shapeEquals(exprShape(X), matrix(3, 4))).toBe(true);
  });

  it('creates variable with options', () => {
    const x = variable(5, { name: 'x', nonneg: true });
    expect(x.data.kind).toBe('variable');
    if (x.data.kind === 'variable') {
      expect(x.data.name).toBe('x');
      expect(x.data.nonneg).toBe(true);
      expect(x.data.nonpos).toBeUndefined();
    }
  });

  it('generates unique IDs', () => {
    const x = variable(5);
    const y = variable(5);
    if (x.data.kind === 'variable' && y.data.kind === 'variable') {
      expect(x.data.id).not.toBe(y.data.id);
    }
  });
});

describe('Constant', () => {
  beforeEach(() => {
    resetExprIds();
  });

  it('creates scalar constant', () => {
    const c = constant(5);
    expect(isConstant(c)).toBe(true);
    if (c.data.kind === 'constant') {
      expect(c.data.value.type).toBe('scalar');
      if (c.data.value.type === 'scalar') {
        expect(c.data.value.value).toBe(5);
      }
    }
  });

  it('creates vector constant', () => {
    const c = constant([1, 2, 3]);
    expect(isConstant(c)).toBe(true);
    expect(shapeEquals(exprShape(c), vector(3))).toBe(true);
  });

  it('creates matrix constant', () => {
    const c = constant([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    expect(isConstant(c)).toBe(true);
    expect(shapeEquals(exprShape(c), matrix(3, 2))).toBe(true);
  });

  it('creates zeros', () => {
    const z = zeros(5);
    expect(isConstant(z)).toBe(true);
    expect(shapeEquals(exprShape(z), vector(5))).toBe(true);

    const Z = zeros(3, 4);
    expect(shapeEquals(exprShape(Z), matrix(3, 4))).toBe(true);
  });

  it('creates ones', () => {
    const o = ones(5);
    expect(isConstant(o)).toBe(true);
    expect(shapeEquals(exprShape(o), vector(5))).toBe(true);
  });

  it('creates identity matrix', () => {
    const I = eye(3);
    expect(isConstant(I)).toBe(true);
    expect(shapeEquals(exprShape(I), matrix(3, 3))).toBe(true);
  });
});

describe('Expression utilities', () => {
  beforeEach(() => {
    resetExprIds();
  });

  describe('exprShape', () => {
    it('returns correct shape for add', () => {
      const x = variable(5);
      const y = variable(5);
      const z = add(x, y);
      expect(shapeEquals(exprShape(z), vector(5))).toBe(true);
    });

    it('returns correct shape for neg', () => {
      const x = variable(5);
      const y = neg(x);
      expect(shapeEquals(exprShape(y), vector(5))).toBe(true);
    });

    it('returns correct shape for sum', () => {
      const x = variable(5);
      const s = sum(x);
      expect(shapeEquals(exprShape(s), scalar())).toBe(true);
    });

    it('returns correct shape for matmul', () => {
      const A = constant([
        [1, 2, 3],
        [4, 5, 6],
      ]); // 2x3
      const x = variable(3);
      const y = matmul(A, x);
      expect(shapeEquals(exprShape(y), vector(2))).toBe(true);
    });

    it('returns correct shape for transpose', () => {
      const A = matrixVar(3, 4);
      const At = transpose(A);
      expect(shapeEquals(exprShape(At), matrix(4, 3))).toBe(true);
    });
  });

  describe('exprVariables', () => {
    it('returns empty set for constant', () => {
      const c = constant(5);
      expect(exprVariables(c).size).toBe(0);
    });

    it('returns variable ID for variable', () => {
      const x = variable(5);
      const vars = exprVariables(x);
      expect(vars.size).toBe(1);
      if (x.data.kind === 'variable') {
        expect(vars.has(x.data.id)).toBe(true);
      }
    });

    it('collects all variables from expression', () => {
      const x = variable(5);
      const y = variable(5);
      const z = add(x, y);
      const vars = exprVariables(z);
      expect(vars.size).toBe(2);
    });
  });

  describe('isConstantExpr', () => {
    it('returns true for constant expressions', () => {
      expect(isConstantExpr(constant(5))).toBe(true);
      expect(isConstantExpr(add(constant(1), constant(2)))).toBe(true);
    });

    it('returns false for expressions with variables', () => {
      const x = variable(5);
      expect(isConstantExpr(x)).toBe(false);
      expect(isConstantExpr(add(x, constant(1)))).toBe(false);
    });
  });
});
