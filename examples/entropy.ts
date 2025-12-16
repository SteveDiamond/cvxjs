/**
 * Maximum Entropy and Logarithmic Examples
 *
 * Demonstrates the exp, log, and entropy atoms for problems involving
 * information theory and exponential cone optimization.
 */

import {
  variable,
  isVariable,
  constant,
  sum,
  mul,
  ge,
  le,
  eq,
  Problem,
  log,
  exp,
  entropy,
} from '../src/index.js';

async function entropyExamples() {
  console.log('=== Entropy and Logarithmic Optimization ===\n');

  // =====================================================
  // Example 1: Maximum Entropy Distribution
  // =====================================================
  console.log('--- Example 1: Maximum Entropy Distribution ---');
  console.log('Find probability distribution p that maximizes entropy');
  console.log('maximize sum(-p * log(p)) subject to sum(p) = 1, p >= 0.01\n');

  {
    const n = 4;
    const p = variable(n);

    // Entropy: -sum(p_i * log(p_i))
    // Our entropy atom computes -x*log(x) element-wise
    const solution = await Problem.maximize(sum(entropy(p)))
      .subjectTo([
        eq(sum(p), constant(1)), // Probabilities sum to 1
        ge(p, constant(new Float64Array(n).fill(0.01))), // p >= 0.01 (for numerical stability)
        le(p, constant(new Float64Array(n).fill(1))), // p <= 1
      ])
      .solve();

    console.log('Status:', solution.status);
    console.log('Maximum entropy:', solution.value?.toFixed(4));
    console.log('Expected (log(4)):', Math.log(4).toFixed(4));

    if (solution.primal && isVariable(p)) {
      const probs = solution.primal.get(p.id)!;
      console.log('\nOptimal distribution (should be uniform):');
      for (let i = 0; i < n; i++) {
        console.log(`  p[${i}] = ${probs[i].toFixed(4)}`);
      }
    }
    console.log();
  }

  // =====================================================
  // Example 2: Log-Barrier (Analytic Center)
  // =====================================================
  console.log('--- Example 2: Analytic Center ---');
  console.log('Find point that maximizes sum(log(x)) in box [0.1, 2]^n');
  console.log('This finds the "analytic center" of the box.\n');

  {
    const n = 3;
    const x = variable(n);
    const lowerBound = 0.1;
    const upperBound = 2.0;

    // Maximize sum(log(x)) + sum(log(upper - x))
    // For simplicity, just maximize sum(log(x)) with box constraints
    const solution = await Problem.maximize(sum(log(x)))
      .subjectTo([
        ge(x, constant(new Float64Array(n).fill(lowerBound))),
        le(x, constant(new Float64Array(n).fill(upperBound))),
      ])
      .solve();

    console.log('Status:', solution.status);
    console.log('Objective:', solution.value?.toFixed(4));

    if (solution.primal && isVariable(x)) {
      const vals = solution.primal.get(x.id)!;
      console.log('\nAnalytic center (should be at upper bound):');
      for (let i = 0; i < n; i++) {
        console.log(`  x[${i}] = ${vals[i].toFixed(4)}`);
      }
    }
    console.log();
  }

  // =====================================================
  // Example 3: Exponential Constraint
  // =====================================================
  console.log('--- Example 3: Exponential Constraint ---');
  console.log('maximize sum(x) subject to exp(x) <= [e, e^2, e^3]\n');

  {
    const n = 3;
    const x = variable(n);
    const bounds = new Float64Array([Math.E, Math.E ** 2, Math.E ** 3]);

    // exp(x) <= bounds means x <= log(bounds) = [1, 2, 3]
    // Maximizing sum(x) with upper bounds should give x = [1, 2, 3]
    const solution = await Problem.maximize(sum(x))
      .subjectTo([le(exp(x), constant(bounds))])
      .solve();

    console.log('Status:', solution.status);
    console.log('Optimal value:', solution.value?.toFixed(4));
    console.log('Expected: 1 + 2 + 3 = 6');

    if (solution.primal && isVariable(x)) {
      const vals = solution.primal.get(x.id)!;
      console.log('\nOptimal x (should be [1, 2, 3]):');
      for (let i = 0; i < n; i++) {
        console.log(`  x[${i}] = ${vals[i].toFixed(4)}`);
      }
    }
    console.log();
  }

  // =====================================================
  // Example 4: Kullback-Leibler Minimization
  // =====================================================
  console.log('--- Example 4: Distribution Matching ---');
  console.log('Find distribution q closest to uniform that satisfies E[x] = 2.5');
  console.log('(using entropy maximization as a proxy)\n');

  {
    const n = 4;
    const outcomes = [1, 2, 3, 4]; // Possible outcomes
    const targetMean = 2.5;

    const q = variable(n);
    const outcomesConst = constant(new Float64Array(outcomes));

    // Maximize entropy subject to E[x] = target and sum(q) = 1
    const solution = await Problem.maximize(sum(entropy(q)))
      .subjectTo([
        eq(sum(mul(outcomesConst, q)), constant(targetMean)), // E[x] = 2.5
        eq(sum(q), constant(1)), // Probabilities sum to 1
        ge(q, constant(new Float64Array(n).fill(0.001))), // q > 0
      ])
      .solve();

    console.log('Status:', solution.status);
    console.log('Entropy:', solution.value?.toFixed(4));

    if (solution.primal && isVariable(q)) {
      const probs = solution.primal.get(q.id)!;
      console.log('\nOptimal distribution:');
      let mean = 0;
      for (let i = 0; i < n; i++) {
        console.log(`  P(X=${outcomes[i]}) = ${probs[i].toFixed(4)}`);
        mean += probs[i] * outcomes[i];
      }
      console.log(`\nComputed mean: ${mean.toFixed(4)} (target: ${targetMean})`);
    }
    console.log();
  }

  console.log('=== Entropy examples complete ===\n');
}

entropyExamples().catch(console.error);
