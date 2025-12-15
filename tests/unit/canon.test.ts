import { describe, it, expect, beforeEach } from 'vitest';
import { variable, constant, resetExprIds } from '../../src/index.js';
import {
  add,
  sum,
  neg,
  mul,
  norm2,
  norm1,
  abs,
  pos,
  maximum,
  minimum,
  index,
  hstack,
} from '../../src/atoms/index.js';
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
    expect(summed.constant[0]).toBe(6); // 1 + 2 + 3
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
      expect(canonicalizer.getConstraints().length).toBe(2); // t >= x and t >= -x
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
      expect(canonicalizer.getConstraints().length).toBe(2); // t >= x and t >= y
    });

    it('canonicalizes minimum', () => {
      const x = variable(5);
      const y = variable(5);
      const m = minimum(x, y);

      const canonicalizer = new Canonicalizer();
      const result = canonicalizer.canonicalize(m);

      expect(result.rows).toBe(5);
      expect(canonicalizer.getConstraints().length).toBe(2); // x >= t and y >= t
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

    const varMap = buildVariableMap(new Set([x.id]), varSizes as Map<number, number>, auxVars);

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

describe('index canonicalization', () => {
  beforeEach(() => {
    resetExprIds();
  });

  it('canonicalizes vector single index', () => {
    // x[2] where x is 5-element vector
    const x = variable(5);
    const y = index(x, 2);

    const canonicalizer = new Canonicalizer();
    const result = canonicalizer.canonicalize(y);

    expect(result.rows).toBe(1); // scalar output
    expect(result.coeffs.size).toBe(1);

    // Verify the coefficient matrix selects element 2
    if (x.kind !== 'variable') throw new Error('Expected variable');
    const coeff = result.coeffs.get(x.id);
    expect(coeff).toBeDefined();
    expect(coeff!.nrows).toBe(1);
    expect(coeff!.ncols).toBe(5);
    // Should have a 1 at column 2
    expect(coeff!.values[0]).toBe(1);
    expect(coeff!.rowIdx[0]).toBe(0);
    // colPtr should indicate the 1 is in column 2
  });

  it('canonicalizes vector range', () => {
    // x[1:4] where x is 5-element vector
    const x = variable(5);
    const y = index(x, [1, 4]);

    const canonicalizer = new Canonicalizer();
    const result = canonicalizer.canonicalize(y);

    expect(result.rows).toBe(3); // 3 elements
    expect(result.coeffs.size).toBe(1);
  });

  it('canonicalizes vector all', () => {
    // x[:] should be identity
    const x = variable(5);
    const y = index(x, 'all');

    const canonicalizer = new Canonicalizer();
    const result = canonicalizer.canonicalize(y);

    expect(result.rows).toBe(5); // same size
    expect(result.coeffs.size).toBe(1);
  });

  it('canonicalizes matrix row extraction', () => {
    // A[0, :] where A is 3x4 matrix variable
    const A = variable([3, 4]);
    const row = index(A, 0, 'all');

    const canonicalizer = new Canonicalizer();
    const result = canonicalizer.canonicalize(row);

    // Row 0 has 4 elements (one from each column)
    expect(result.rows).toBe(4);
    expect(result.coeffs.size).toBe(1);
  });

  it('canonicalizes matrix column extraction', () => {
    // A[:, 1] where A is 3x4 matrix variable
    const A = variable([3, 4]);
    const col = index(A, 'all', 1);

    const canonicalizer = new Canonicalizer();
    const result = canonicalizer.canonicalize(col);

    // Column 1 has 3 elements
    expect(result.rows).toBe(3);
    expect(result.coeffs.size).toBe(1);
  });

  it('canonicalizes matrix single element', () => {
    // A[1, 2] where A is 3x4 matrix variable
    const A = variable([3, 4]);
    const elem = index(A, 1, 2);

    const canonicalizer = new Canonicalizer();
    const result = canonicalizer.canonicalize(elem);

    expect(result.rows).toBe(1); // scalar
    expect(result.coeffs.size).toBe(1);
  });

  it('canonicalizes matrix submatrix', () => {
    // A[0:2, 1:3] where A is 3x4 matrix variable
    const A = variable([3, 4]);
    const sub = index(A, [0, 2], [1, 3]);

    const canonicalizer = new Canonicalizer();
    const result = canonicalizer.canonicalize(sub);

    // 2x2 submatrix = 4 elements
    expect(result.rows).toBe(4);
    expect(result.coeffs.size).toBe(1);
  });

  it('canonicalizes nested index expressions', () => {
    // (x + y)[1:3] where x, y are 5-element vectors
    const x = variable(5);
    const y = variable(5);
    const z = add(x, y);
    const slice = index(z, [1, 3]);

    const canonicalizer = new Canonicalizer();
    const result = canonicalizer.canonicalize(slice);

    expect(result.rows).toBe(2);
    expect(result.coeffs.size).toBe(2); // x and y coefficients
  });
});

describe('hstack canonicalization', () => {
  beforeEach(() => {
    resetExprIds();
  });

  it('canonicalizes hstack of vectors', () => {
    const x = variable(3);
    const y = variable(3);
    const z = hstack(x, y);

    const canonicalizer = new Canonicalizer();
    const result = canonicalizer.canonicalize(z);

    // hstack of two 3-vectors treated as 3x1 columns -> 3x2 matrix = 6 elements
    expect(result.rows).toBe(6);
    expect(result.coeffs.size).toBe(2);
  });

  it('canonicalizes hstack of matrices', () => {
    const A = variable([2, 3]);
    const B = variable([2, 4]);
    const C = hstack(A, B);

    const canonicalizer = new Canonicalizer();
    const result = canonicalizer.canonicalize(C);

    // 2x3 hstack 2x4 = 2x7 = 14 elements
    expect(result.rows).toBe(14);
    expect(result.coeffs.size).toBe(2);
  });

  it('canonicalizes hstack with constants', () => {
    const x = variable(3);
    const c = constant([1, 2, 3]);
    const z = hstack(x, c);

    const canonicalizer = new Canonicalizer();
    const result = canonicalizer.canonicalize(z);

    // Two 3-vectors stacked horizontally = 3x2 = 6 elements
    expect(result.rows).toBe(6);
    expect(result.coeffs.size).toBe(1); // only x has coefficients

    // Constant part should have [0,0,0, 1,2,3]
    expect(result.constant[3]).toBe(1);
    expect(result.constant[4]).toBe(2);
    expect(result.constant[5]).toBe(3);
  });

  it('canonicalizes hstack with expressions', () => {
    const x = variable(3);
    const y = variable(3);
    const z = hstack(add(x, y), neg(x));

    const canonicalizer = new Canonicalizer();
    const result = canonicalizer.canonicalize(z);

    expect(result.rows).toBe(6);
    expect(result.coeffs.size).toBe(2); // x and y coefficients
  });
});
