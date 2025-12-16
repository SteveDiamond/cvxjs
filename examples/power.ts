/**
 * Power and Square Root Examples
 *
 * Demonstrates the sqrt and power atoms for problems involving
 * power cone optimization.
 */

import { variable, Problem } from '../src/index.js';

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

    const solution = await Problem.maximize(x.sqrt().sum())
      .subjectTo([
        x.sum().eq(budget), // Budget constraint
        x.ge(0.001), // x > 0
      ])
      .solve();

    console.log('Status:', solution.status);
    console.log('Maximum sum(sqrt(x)):', solution.value?.toFixed(4));
    console.log('Expected (4 * sqrt(1) = 4):', (4 * Math.sqrt(1)).toFixed(4));

    const vals = solution.valueOf(x)!;
    console.log('\nOptimal allocation (should be uniform):');
    for (let i = 0; i < n; i++) {
      console.log(`  x[${i}] = ${vals[i].toFixed(4)} (sqrt = ${Math.sqrt(vals[i]).toFixed(4)})`);
    }
    console.log();
  }

  // =====================================================
  // Example 2: Power Function (x^0.5)
  // =====================================================
  console.log('--- Example 2: Power Function (x^0.5) ---');
  console.log('maximize sum(x^0.5) subject to sum(x) = 9, x >= 0\n');

  {
    const n = 3;
    const x = variable(n);

    const solution = await Problem.maximize(x.power(0.5).sum())
      .subjectTo([
        x.sum().eq(9),
        x.ge(0.001),
      ])
      .solve();

    console.log('Status:', solution.status);
    console.log('Maximum sum(x^0.5):', solution.value?.toFixed(4));
    console.log('Expected (3 * sqrt(3)):', (3 * Math.sqrt(3)).toFixed(4));

    const vals = solution.valueOf(x)!;
    console.log('\nOptimal x (should be [3, 3, 3]):');
    for (let i = 0; i < n; i++) {
      console.log(`  x[${i}] = ${vals[i].toFixed(4)}`);
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

    const solution = await Problem.minimize(x.sumSquares())
      .subjectTo([
        x.sum().eq(6),
        x.ge(0),
      ])
      .solve();

    console.log('Status:', solution.status);
    console.log('Minimum sum(x^2):', solution.value?.toFixed(4));
    console.log('Expected (3 * 4 = 12):', 12);

    const vals = solution.valueOf(x)!;
    console.log('\nOptimal x (should be [2, 2, 2]):');
    for (let i = 0; i < n; i++) {
      console.log(`  x[${i}] = ${vals[i].toFixed(4)}`);
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

    const solution = await Problem.minimize(x.power(2).sum())
      .subjectTo([
        x.sum().ge(3),
        x.ge(0),
      ])
      .solve();

    console.log('Status:', solution.status);
    console.log('Minimum sum(x^2):', solution.value?.toFixed(4));
    console.log('Expected (3 * 1 = 3):', 3);

    const vals = solution.valueOf(x)!;
    console.log('\nOptimal x (should be [1, 1, 1]):');
    for (let i = 0; i < n; i++) {
      console.log(`  x[${i}] = ${vals[i].toFixed(4)}`);
    }
    console.log();
  }

  // =====================================================
  // Example 5: Utility Maximization with Diminishing Returns
  // =====================================================
  console.log('--- Example 5: Resource Allocation with Diminishing Returns ---');
  console.log('maximize sum(x_i^0.3) subject to sum(x) = 10, x >= 0\n');

  {
    const n = 4;
    const x = variable(n);

    const solution = await Problem.maximize(x.power(0.3).sum())
      .subjectTo([
        x.sum().eq(10),
        x.ge(0.01),
      ])
      .solve();

    console.log('Status:', solution.status);
    console.log('Maximum utility:', solution.value?.toFixed(4));

    const vals = solution.valueOf(x)!;
    console.log('\nOptimal allocation (equal weights -> uniform):');
    for (let i = 0; i < n; i++) {
      const utility = Math.pow(vals[i], 0.3);
      console.log(`  x[${i}] = ${vals[i].toFixed(4)} (utility = ${utility.toFixed(4)})`);
    }
    console.log();
  }

  console.log('=== Power examples complete ===\n');
}

powerExamples().catch(console.error);
