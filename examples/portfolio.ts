/**
 * Portfolio Optimization Example
 *
 * Classic mean-variance portfolio optimization (Markowitz model).
 * Minimize portfolio risk (variance) subject to a minimum return constraint.
 *
 * Problem:
 *   minimize    w' Σ w           (portfolio variance)
 *   subject to  μ' w >= r_min    (minimum return)
 *               1' w = 1         (weights sum to 1)
 *               w >= 0           (no short selling)
 *
 * where:
 *   w = portfolio weights
 *   Σ = covariance matrix of returns
 *   μ = expected returns
 *   r_min = minimum required return
 */

import {
  variable,
  constant,
  sum,
  mul,
  ge,
  eq,
  Problem,
  quadForm,
  add,
} from '../src/index.js';

async function portfolioOptimization() {
  console.log('=== Portfolio Optimization (Mean-Variance) ===\n');

  // 4 assets with expected returns and covariance matrix
  const expectedReturns = [0.12, 0.1, 0.07, 0.03]; // 12%, 10%, 7%, 3%

  // Covariance matrix (symmetric positive definite) - row-major as 2D array
  // Asset 1: 20% vol, Asset 2: 15.8% vol, Asset 3: 10% vol, Asset 4: 5% vol
  const covariance = [
    [0.04, 0.006, 0.002, 0.0],
    [0.006, 0.025, 0.004, 0.001],
    [0.002, 0.004, 0.01, 0.002],
    [0.0, 0.001, 0.002, 0.0025],
  ];

  const n = 4;
  const minReturn = 0.08; // Require at least 8% expected return
  const riskAversion = 2.0; // Risk aversion parameter

  // Decision variable: portfolio weights
  const w = variable(n);

  // Covariance matrix and expected returns as constants
  const Sigma = constant(covariance);
  const mu = constant(new Float64Array(expectedReturns));

  // =====================================================
  // Example 1: Minimize variance subject to return constraint
  // =====================================================
  console.log('--- Example 1: Minimum Variance Portfolio ---');
  console.log(`minimize w'Σw subject to μ'w >= ${minReturn * 100}%, sum(w) = 1, w >= 0\n`);

  {
    // Objective: minimize w' Σ w (portfolio variance)
    // Using quadForm for native QP support!
    const solution = await Problem.minimize(quadForm(w, Sigma))
      .subjectTo([
        ge(sum(mul(mu, w)), constant(minReturn)), // Return >= min return
        eq(sum(w), constant(1)), // Weights sum to 1
        ge(w, constant(new Float64Array(n))), // No short selling
      ])
      .solve();

    console.log('Status:', solution.status);
    console.log('Portfolio variance:', solution.value?.toFixed(6));
    console.log('Portfolio std dev:', Math.sqrt(solution.value!).toFixed(4), '(' + (Math.sqrt(solution.value!) * 100).toFixed(2) + '%)');

    if (solution.primal) {
      const weights = solution.primal.values().next().value as Float64Array;
      console.log('\nOptimal weights:');
      const assetNames = ['Stock A (12%)', 'Stock B (10%)', 'Stock C (7%)', 'Bonds (3%)'];
      let expectedReturn = 0;
      for (let i = 0; i < n; i++) {
        console.log(`  ${assetNames[i]}: ${(weights[i] * 100).toFixed(1)}%`);
        expectedReturn += weights[i] * expectedReturns[i];
      }
      console.log(`\nExpected return: ${(expectedReturn * 100).toFixed(2)}%`);
    }
    console.log();
  }

  // =====================================================
  // Example 2: Mean-Variance Tradeoff (Risk Aversion)
  // =====================================================
  console.log('--- Example 2: Mean-Variance Tradeoff ---');
  console.log(`minimize (${riskAversion} * w'Σw - μ'w) subject to sum(w) = 1, w >= 0\n`);

  {
    // Objective: minimize λ * variance - return
    // This is a risk-aversion weighted tradeoff
    const variance = quadForm(w, Sigma);
    const returnTerm = sum(mul(mu, w));

    const solution = await Problem.minimize(add(mul(riskAversion, variance), mul(-1, returnTerm)))
      .subjectTo([
        eq(sum(w), constant(1)), // Weights sum to 1
        ge(w, constant(new Float64Array(n))), // No short selling
      ])
      .solve();

    console.log('Status:', solution.status);

    if (solution.primal) {
      const weights = solution.primal.values().next().value as Float64Array;
      console.log('\nOptimal weights:');
      const assetNames = ['Stock A (12%)', 'Stock B (10%)', 'Stock C (7%)', 'Bonds (3%)'];
      let expectedReturn = 0;
      let portfolioVariance = 0;
      for (let i = 0; i < n; i++) {
        console.log(`  ${assetNames[i]}: ${(weights[i] * 100).toFixed(1)}%`);
        expectedReturn += weights[i] * expectedReturns[i];
      }
      // Compute variance: w' Σ w
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          portfolioVariance += weights[i] * weights[j] * covariance[i][j];
        }
      }
      console.log(`\nExpected return: ${(expectedReturn * 100).toFixed(2)}%`);
      console.log(`Portfolio std dev: ${(Math.sqrt(portfolioVariance) * 100).toFixed(2)}%`);
    }
    console.log();
  }

  console.log('=== Portfolio optimization complete ===\n');
}

portfolioOptimization().catch(console.error);
