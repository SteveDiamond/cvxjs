/**
 * Basic Usage Example
 *
 * Demonstrates the core cvxjs API with simple optimization problems.
 */

import { variable, Problem } from '../src/index.js';

async function basicExamples() {
  console.log('=== cvxjs Basic Examples ===\n');

  // Example 1: Simple LP
  console.log('--- Example 1: Linear Program ---');
  console.log('minimize sum(x) subject to x >= 1\n');

  {
    const x = variable(3);

    const solution = await Problem.minimize(x.sum())
      .subjectTo([x.ge(1)])
      .solve();

    console.log('Status:', solution.status);
    console.log('Optimal value:', solution.value);
    console.log('x =', Array.from(solution.valueOf(x)!).map((v) => v.toFixed(2)));
    console.log();
  }

  // Example 2: LP with bounds
  console.log('--- Example 2: Bounded LP ---');
  console.log('maximize sum(x) subject to 0 <= x <= 2\n');

  {
    const x = variable(3);

    const solution = await Problem.maximize(x.sum())
      .subjectTo([x.ge(0), x.le(2)])
      .solve();

    console.log('Status:', solution.status);
    console.log('Optimal value:', solution.value);
    console.log('x =', Array.from(solution.valueOf(x)!).map((v) => v.toFixed(2)));
    console.log();
  }

  // Example 3: SOCP with norm2
  console.log('--- Example 3: Second-Order Cone Program ---');
  console.log('minimize ||x||_2 subject to sum(x) = 3\n');

  {
    const x = variable(3);

    const solution = await Problem.minimize(x.norm2())
      .subjectTo([x.sum().eq(3)])
      .solve();

    console.log('Status:', solution.status);
    console.log('Optimal ||x||_2:', solution.value?.toFixed(4));
    console.log('Expected: sqrt(3) =', Math.sqrt(3).toFixed(4));
    console.log('x =', Array.from(solution.valueOf(x)!).map((v) => v.toFixed(4)));
    console.log();
  }

  // Example 4: L1 minimization
  console.log('--- Example 4: L1 Norm Minimization ---');
  console.log('minimize ||x||_1 subject to sum(x) = 3, x >= 0\n');

  {
    const x = variable(3);

    const solution = await Problem.minimize(x.norm1())
      .subjectTo([x.sum().eq(3), x.ge(0)])
      .solve();

    console.log('Status:', solution.status);
    console.log('Optimal ||x||_1:', solution.value?.toFixed(4));
    console.log('x =', Array.from(solution.valueOf(x)!).map((v) => v.toFixed(4)));
    console.log();
  }

  console.log('=== All examples completed ===\n');
}

basicExamples().catch(console.error);
