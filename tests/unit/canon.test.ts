import { describe, it, expect, beforeEach } from 'vitest';
import {
  variable,
  constant,
  resetExprIds,
  exprVariables,
  exprShape,
} from '../../src/index.js';
import { add, sum, neg, mul, norm2, norm1, abs, pos, maximum, minimum } from '../../src/atoms/index.js';
import {
  linExprVariable,
  linExprConstant,
  linExprAdd,
  linExprNeg,
  linExprScale,
  linExprSum,
  linExprIsConstant,
  Canonicalizer,
  canonicalizeProblem,
  buildVariableMap,
  stuffProblem,
} from '../../src/canon/index.js';
import { size } from '../../src/expr/shape.js';

describe('LinExpr', () => {
  beforeEach(() => {
    resetExprIds();
  });

  it('creates variable LinExpr', () => {
    const x = variable(5);
    if (x.kind !== 'variable') throw new Error('Expected variable');

    const lin = linExprVariable(x.id, 5);
    expect(lin.rows).toBe(5);
    expect(lin.coeffs.size).toBe(1);
    expect(lin.coeffs.has(x.id)).toBe(true);
  });

  it('creates constant LinExpr', () => {
    const c = new Float64Array([1, 2, 3]);
    const lin = linExprConstant(c);
    expect(lin.rows).toBe(3);
    expect(lin.coeffs.size).toBe(0);
    expect(linExprIsConstant(lin)).toBe(true);
  });

  it('adds LinExprs', () => {
    const x = variable(5);
    const y = variable(5);
    if (x.kind !== 'variable' || y.kind !== 'variable') throw new Error('Expected variables');

    const linX = linExprVariable(x.id, 5);
    const linY = linExprVariable(y.id, 5);
    const sum = linExprAdd(linX, linY);

    expect(sum.rows).toBe(5);
    expect(sum.coeffs.size).toBe(2);
  });

  it('negates LinExpr', () => {
    const c = new Float64Array([1, 2, 3]);
    const lin = linExprConstant(c);
    const negLin = linExprNeg(lin);

    expect(negLin.constant[0]).toBe(-1);
    expect(negLin.constant[1]).toBe(-2);
    expect(negLin.constant[2]).toBe(-3);
  });

  it('scales LinExpr', () => {
    const c = new Float64Array([1, 2, 3]);
    const lin = linExprConstant(c);
    const scaled = linExprScale(lin, 2);

    expect(scaled.constant[0]).toBe(2);
    expect(scaled.constant[1]).toBe(4);
    expect(scaled.constant[2]).toBe(6);
  });

  it('sums LinExpr elements', () => {
    const c = new Float64Array([1, 2, 3]);
    const lin = linExprConstant(c);
    const summed = linExprSum(lin);

    expect(summed.rows).toBe(1);
    expect(summed.constant[0]).toBe(6);  // 1 + 2 + 3
  });
});

describe('Canonicalizer', () => {
  beforeEach(() => {
    resetExprIds();
  });

  describe('affine expressions', () => {
    it('canonicalizes variable', () => {
      const x = variable(5);
      const canonicalizer = new Canonicalizer();
      const result = canonicalizer.canonicalize(x);

      expect(result.rows).toBe(5);
      expect(result.coeffs.size).toBe(1);
    });

    it('canonicalizes constant', () => {
      const c = constant([1, 2, 3]);
      const canonicalizer = new Canonicalizer();
      const result = canonicalizer.canonicalize(c);

      expect(result.rows).toBe(3);
      expect(linExprIsConstant(result)).toBe(true);
    });

    it('canonicalizes add', () => {
      const x = variable(5);
      const y = variable(5);
      const z = add(x, y);

      const canonicalizer = new Canonicalizer();
      const result = canonicalizer.canonicalize(z);

      expect(result.rows).toBe(5);
      expect(result.coeffs.size).toBe(2);
    });

    it('canonicalizes neg', () => {
      const x = variable(5);
      const y = neg(x);

      const canonicalizer = new Canonicalizer();
      const result = canonicalizer.canonicalize(y);

      expect(result.rows).toBe(5);
    });

    it('canonicalizes sum', () => {
      const x = variable(5);
      const s = sum(x);

      const canonicalizer = new Canonicalizer();
      const result = canonicalizer.canonicalize(s);

      expect(result.rows).toBe(1);
    });

    it('canonicalizes scalar multiplication', () => {
      const x = variable(5);
      const y = mul(constant(2), x);

      const canonicalizer = new Canonicalizer();
      const result = canonicalizer.canonicalize(y);

      expect(result.rows).toBe(5);
    });
  });

  describe('nonlinear expressions', () => {
    it('canonicalizes norm2', () => {
      const x = variable(5);
      const n = norm2(x);

      const canonicalizer = new Canonicalizer();
      const result = canonicalizer.canonicalize(n);

      // norm2 introduces SOC constraint
      expect(result.rows).toBe(1);
      expect(canonicalizer.getConstraints().length).toBe(1);
      expect(canonicalizer.getConstraints()[0]!.kind).toBe('soc');
      expect(canonicalizer.getAuxVars().length).toBe(1);
    });

    it('canonicalizes norm1', () => {
      const x = variable(5);
      const n = norm1(x);

      const canonicalizer = new Canonicalizer();
      const result = canonicalizer.canonicalize(n);

      // norm1 introduces inequality constraints
      expect(result.rows).toBe(1);
      expect(canonicalizer.getConstraints().length).toBe(2);  // t >= x and t >= -x
      expect(canonicalizer.getAuxVars().length).toBe(1);
    });

    it('canonicalizes abs', () => {
      const x = variable(5);
      const a = abs(x);

      const canonicalizer = new Canonicalizer();
      const result = canonicalizer.canonicalize(a);

      expect(result.rows).toBe(5);
      expect(canonicalizer.getConstraints().length).toBe(2);
    });

    it('canonicalizes pos', () => {
      const x = variable(5);
      const p = pos(x);

      const canonicalizer = new Canonicalizer();
      const result = canonicalizer.canonicalize(p);

      expect(result.rows).toBe(5);
      expect(canonicalizer.getConstraints().length).toBe(1);
    });

    it('canonicalizes maximum', () => {
      const x = variable(5);
      const y = variable(5);
      const m = maximum(x, y);

      const canonicalizer = new Canonicalizer();
      const result = canonicalizer.canonicalize(m);

      expect(result.rows).toBe(5);
      expect(canonicalizer.getConstraints().length).toBe(2);  // t >= x and t >= y
    });

    it('canonicalizes minimum', () => {
      const x = variable(5);
      const y = variable(5);
      const m = minimum(x, y);

      const canonicalizer = new Canonicalizer();
      const result = canonicalizer.canonicalize(m);

      expect(result.rows).toBe(5);
      expect(canonicalizer.getConstraints().length).toBe(2);  // x >= t and y >= t
    });
  });
});

describe('canonicalizeProblem', () => {
  beforeEach(() => {
    resetExprIds();
  });

  it('canonicalizes simple LP', () => {
    const x = variable(5);
    const obj = sum(x);

    const result = canonicalizeProblem(obj, [], 'minimize');

    expect(result.objectiveLinExpr.rows).toBe(1);
    expect(result.coneConstraints.length).toBe(0);
    expect(result.auxVars.length).toBe(0);
  });

  it('canonicalizes problem with norm2 objective', () => {
    const x = variable(5);
    const obj = norm2(x);

    const result = canonicalizeProblem(obj, [], 'minimize');

    expect(result.objectiveLinExpr.rows).toBe(1);
    expect(result.coneConstraints.length).toBe(1);
    expect(result.auxVars.length).toBe(1);
  });
});

describe('stuffProblem', () => {
  beforeEach(() => {
    resetExprIds();
  });

  it('stuffs simple LP', () => {
    const x = variable(5);
    if (x.kind !== 'variable') throw new Error('Expected variable');

    const obj = sum(x);
    const { objectiveLinExpr, coneConstraints, auxVars } = canonicalizeProblem(obj, [], 'minimize');

    const varSizes = new Map<number, number>();
    varSizes.set(x.id as number, 5);

    const varMap = buildVariableMap(
      new Set([x.id]),
      varSizes as Map<any, number>,
      auxVars
    );

    const stuffed = stuffProblem(objectiveLinExpr, coneConstraints, varMap, 0);

    expect(stuffed.nVars).toBe(5);
    expect(stuffed.nConstraints).toBe(0);
    expect(stuffed.q.length).toBe(5);

    // q should be all 1s (coefficient of sum)
    for (let i = 0; i < 5; i++) {
      expect(stuffed.q[i]).toBe(1);
    }
  });
});
