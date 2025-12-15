import { describe, it, expect } from 'vitest';
import {
  loadWasm,
  testWasm,
  clarabelVersion,
  solveConic,
  cscFromTriplets,
  cscZeros,
} from '../../src/index.js';
import type { ConeDims } from '../../src/canon/index.js';

describe('Direct WASM Tests', () => {
  it('loads WASM successfully', async () => {
    const result = await testWasm();
    expect(result).toBe('Clarabel WASM is working!');
  });

  it('gets Clarabel version', async () => {
    const version = await clarabelVersion();
    expect(version).toContain('clarabel-wasm');
  });

  it('solves trivial LP: min 0 s.t. nothing', async () => {
    // Most minimal problem: 0 variables, 0 constraints
    const P = cscZeros(0, 0);
    const q = new Float64Array(0);
    const A = cscZeros(0, 0);
    const b = new Float64Array(0);
    const coneDims: ConeDims = {
      zero: 0,
      nonneg: 0,
      soc: [],
      exp: 0,
      power: [],
    };

    // This might fail because Clarabel might not accept empty problems
    // Let's see
    try {
      const result = await solveConic(P, q, A, b, coneDims, {});
      console.log('Empty problem result:', result);
    } catch (e) {
      console.log('Empty problem error:', e);
    }
  });

  it('solves simple 1-variable LP', async () => {
    // min x s.t. x >= 1
    // Becomes: min x s.t. -x + s = -1, s >= 0
    // Or in standard form: min x s.t. -x + s = -1, s in Nonneg
    //
    // Actually Clarabel standard form is: Ax + s = b, s in K
    // So for x >= 1, we have: -x + s = -1, s >= 0
    // A = [-1], b = [-1], K = Nonneg(1)
    //
    // Wait, the standard form is: Ax + s = b, where s is the slack
    // x >= 1 means x - 1 >= 0
    // So we have: x - 1 >= 0
    // In standard form: -x + s = -1, s >= 0
    // A = [[-1]], b = [-1], K = Nonneg(1)

    const n = 1; // 1 variable
    const m = 1; // 1 constraint

    // P = 0 (no quadratic term)
    const P = cscZeros(n, n);

    // q = [1] (minimize x)
    const q = new Float64Array([1]);

    // A = [[-1]]
    const A = cscFromTriplets(m, n, [0], [0], [-1]);

    // b = [-1]
    const b = new Float64Array([-1]);

    const coneDims: ConeDims = {
      zero: 0,
      nonneg: 1,
      soc: [],
      exp: 0,
      power: [],
    };

    console.log('Calling solveConic with:');
    console.log('  P:', P);
    console.log('  q:', q);
    console.log('  A:', A);
    console.log('  b:', b);
    console.log('  coneDims:', coneDims);

    const result = await solveConic(P, q, A, b, coneDims, { verbose: true });

    console.log('Result:', result);

    expect(result.status).toBe('optimal');
    expect(result.objVal).toBeCloseTo(1, 5); // x* = 1
  });

  it('solves 1-variable LP with equality', async () => {
    // min x s.t. x = 2
    // A = [[1]], b = [2], K = Zero(1)

    const n = 1;
    const m = 1;

    const P = cscZeros(n, n);
    const q = new Float64Array([1]);
    const A = cscFromTriplets(m, n, [0], [0], [1]);
    const b = new Float64Array([2]);

    const coneDims: ConeDims = {
      zero: 1,
      nonneg: 0,
      soc: [],
      exp: 0,
      power: [],
    };

    const result = await solveConic(P, q, A, b, coneDims, {});
    console.log('Equality constraint result:', result);

    expect(result.status).toBe('optimal');
    expect(result.objVal).toBeCloseTo(2, 5);
  });
});
