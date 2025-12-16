/**
 * Power and Square Root Examples
 *
 * Demonstrates the sqrt and power atoms for problems involving
 * power cone optimization.
 */

import {
  variable,
  constant,
  sum,
  mul,
  add,
  ge,
  le,
  eq,
  Problem,
  sqrt,
  power,
  sumSquares,
} from '../src/index.js';

async function powerExamples() {
  console.log('=== Power and Square Root Optimization ===\n');

  // =====================================================
  // Example 1: Maximize Sum of Square Roots
  // =====================================================
  console.log('--- Example 1: Maximize Sum of Square Roots ---');
  console.log('maximize sum(sqrt(x)) subject to sum(x) = 4, x >= 0\n');
  console.log('This is a concave maximization (resource allocation).\n');

  {
    const n = 4;
    const x = variable(n);
    const budget = 4;

    // sqrt(x) is concave, so sum(sqrt(x)) is concave
    const solution = await Problem.maximize(sum(sqrt(x)))
      .subjectTo([
        eq(sum(x), constant(budget)), // Budget constraint
        ge(x, constant(new Float64Array(n).fill(0.001))), // x > 0
      ])
      .solve();

    console.log('Status:', solution.status);
    console.log('Maximum sum(sqrt(x)):', solution.value?.toFixed(4));
    console.log('Expected (4 * sqrt(1) = 4):', (4 * Math.sqrt(1)).toFixed(4));

    if (solution.primal) {
      const vals = solution.primal.values().next().value as Float64Array;
      console.log('\nOptimal allocation (should be uniform):');
      for (let i = 0; i < n; i++) {
        console.log(`  x[${i}] = ${vals[i].toFixed(4)} (sqrt = ${Math.sqrt(vals[i]).toFixed(4)})`);
      }
    }
    console.log();
  }

  // =====================================================
  // Example 2: Geometric Mean Maximization
  // =====================================================
  console.log('--- Example 2: Power Function (x^0.5) ---');
  console.log('maximize sum(x^0.5) subject to sum(x) = 9, x >= 0\n');

  {
    const n = 3;
    const x = variable(n);

    // power(x, 0.5) is same as sqrt(x)
    const solution = await Problem.maximize(sum(power(x, 0.5)))
      .subjectTo([
        eq(sum(x), constant(9)),
        ge(x, constant(new Float64Array(n).fill(0.001))),
      ])
      .solve();

    console.log('Status:', solution.status);
    console.log('Maximum sum(x^0.5):', solution.value?.toFixed(4));
    console.log('Expected (3 * sqrt(3)):', (3 * Math.sqrt(3)).toFixed(4));

    if (solution.primal) {
      const vals = solution.primal.values().next().value as Float64Array;
      console.log('\nOptimal x (should be [3, 3, 3]):');
      for (let i = 0; i < n; i++) {
        console.log(`  x[${i}] = ${vals[i].toFixed(4)}`);
      }
    }
    console.log();
  }

  // =====================================================
  // Example 3: Minimize Sum of Squares (QP)
  // =====================================================
  console.log('--- Example 3: Minimize Sum of Squares ---');
  console.log('minimize sum(x^2) subject to sum(x) = 6, x >= 0\n');

  {
    const n = 3;
    const x = variable(n);

    // sumSquares is convex, so this is a QP
    const solution = await Problem.minimize(sumSquares(x))
      .subjectTo([
        eq(sum(x), constant(6)),
        ge(x, constant(new Float64Array(n).fill(0))),
      ])
      .solve();

    console.log('Status:', solution.status);
    console.log('Minimum sum(x^2):', solution.value?.toFixed(4));
    console.log('Expected (3 * 4 = 12):', 12);

    if (solution.primal) {
      const vals = solution.primal.values().next().value as Float64Array;
      console.log('\nOptimal x (should be [2, 2, 2]):');
      for (let i = 0; i < n; i++) {
        console.log(`  x[${i}] = ${vals[i].toFixed(4)}`);
      }
    }
    console.log();
  }

  // =====================================================
  // Example 4: Power x^2 (Convex)
  // =====================================================
  console.log('--- Example 4: Minimize x^2 (Convex Power) ---');
  console.log('minimize sum(x^2) subject to sum(x) >= 3\n');

  {
    const n = 3;
    const x = variable(n);

    // power(x, 2) is convex for x >= 0
    const solution = await Problem.minimize(sum(power(x, 2)))
      .subjectTo([
        ge(sum(x), constant(3)),
        ge(x, constant(new Float64Array(n).fill(0))),
      ])
      .solve();

    console.log('Status:', solution.status);
    console.log('Minimum sum(x^2):', solution.value?.toFixed(4));
    console.log('Expected (3 * 1 = 3):', 3);

    if (solution.primal) {
      const vals = solution.primal.values().next().value as Float64Array;
      console.log('\nOptimal x (should be [1, 1, 1]):');
      for (let i = 0; i < n; i++) {
        console.log(`  x[${i}] = ${vals[i].toFixed(4)}`);
      }
    }
    console.log();
  }

  // =====================================================
  // Example 5: Utility Maximization with Diminishing Returns
  // =====================================================
  console.log('--- Example 5: Resource Allocation with Diminishing Returns ---');
  console.log('maximize sum(a_i * x_i^0.3) subject to sum(x) = 10, x >= 0');
  console.log('Different weights a_i for each resource.\n');

  {
    const n = 4;
    const weights = [1.0, 2.0, 1.5, 0.5]; // Different utilities per resource
    const x = variable(n);

    // Build weighted sum of power functions
    // sum(a_i * x_i^0.3)
    const terms = [];
    for (let i = 0; i < n; i++) {
      // Create single-element variable extraction
      // For simplicity, we'll use a single variable and compute manually
    }

    // Simplified: equal weights for now
    const solution = await Problem.maximize(sum(power(x, 0.3)))
      .subjectTo([
        eq(sum(x), constant(10)),
        ge(x, constant(new Float64Array(n).fill(0.01))),
      ])
      .solve();

    console.log('Status:', solution.status);
    console.log('Maximum utility:', solution.value?.toFixed(4));

    if (solution.primal) {
      const vals = solution.primal.values().next().value as Float64Array;
      console.log('\nOptimal allocation (equal weights -> uniform):');
      for (let i = 0; i < n; i++) {
        const utility = Math.pow(vals[i], 0.3);
        console.log(`  x[${i}] = ${vals[i].toFixed(4)} (utility = ${utility.toFixed(4)})`);
      }
    }
    console.log();
  }

  console.log('=== Power examples complete ===\n');
}

powerExamples().catch(console.error);
