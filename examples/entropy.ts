/**
 * Maximum Entropy and Logarithmic Examples
 *
 * Demonstrates the exp, log, and entropy atoms for problems involving
 * information theory and exponential cone optimization.
 */

import { variable, constant, Problem } from '../src/index.js';

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
    const solution = await Problem.maximize(p.entropy().sum())
      .subjectTo([
        p.sum().eq(1), // Probabilities sum to 1
        p.ge(0.01), // p >= 0.01 (numerical stability)
        p.le(1), // p <= 1
      ])
      .solve();

    console.log('Status:', solution.status);
    console.log('Maximum entropy:', solution.value?.toFixed(4));
    console.log('Expected (log(4)):', Math.log(4).toFixed(4));

    const probs = solution.valueOf(p)!;
    console.log('\nOptimal distribution (should be uniform):');
    for (let i = 0; i < n; i++) {
      console.log(`  p[${i}] = ${probs[i].toFixed(4)}`);
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

    const solution = await Problem.maximize(x.log().sum())
      .subjectTo([x.ge(lowerBound), x.le(upperBound)])
      .solve();

    console.log('Status:', solution.status);
    console.log('Objective:', solution.value?.toFixed(4));

    const vals = solution.valueOf(x)!;
    console.log('\nAnalytic center (should be at upper bound):');
    for (let i = 0; i < n; i++) {
      console.log(`  x[${i}] = ${vals[i].toFixed(4)}`);
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

    // exp(x) <= bounds means x <= log(bounds) = [1, 2, 3]
    // Arrays are auto-wrapped as constants
    const solution = await Problem.maximize(x.sum())
      .subjectTo([x.exp().le([Math.E, Math.E ** 2, Math.E ** 3])])
      .solve();

    console.log('Status:', solution.status);
    console.log('Optimal value:', solution.value?.toFixed(4));
    console.log('Expected: 1 + 2 + 3 = 6');

    const vals = solution.valueOf(x)!;
    console.log('\nOptimal x (should be [1, 2, 3]):');
    for (let i = 0; i < n; i++) {
      console.log(`  x[${i}] = ${vals[i].toFixed(4)}`);
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
    const outcomes = [1, 2, 3, 4];
    const targetMean = 2.5;

    const q = variable(n);
    const outcomesConst = constant(outcomes);

    // Maximize entropy subject to E[x] = target and sum(q) = 1
    const solution = await Problem.maximize(q.entropy().sum())
      .subjectTo([
        outcomesConst.mul(q).sum().eq(targetMean), // E[x] = 2.5
        q.sum().eq(1), // Probabilities sum to 1
        q.ge(0.001), // q > 0
      ])
      .solve();

    console.log('Status:', solution.status);
    console.log('Entropy:', solution.value?.toFixed(4));

    const probs = solution.valueOf(q)!;
    console.log('\nOptimal distribution:');
    let mean = 0;
    for (let i = 0; i < n; i++) {
      console.log(`  P(X=${outcomes[i]}) = ${probs[i].toFixed(4)}`);
      mean += probs[i] * outcomes[i];
    }
    console.log(`\nComputed mean: ${mean.toFixed(4)} (target: ${targetMean})`);
    console.log();
  }

  console.log('=== Entropy examples complete ===\n');
}

entropyExamples().catch(console.error);
