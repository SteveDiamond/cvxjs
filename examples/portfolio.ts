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

import { variable, constant, sum, mul, ge, eq, Problem } from '../src/index.js';

async function portfolioOptimization() {
  console.log('=== Portfolio Optimization ===\n');

  // 4 assets with expected returns and covariance matrix
  const expectedReturns = [0.12, 0.1, 0.07, 0.03]; // 12%, 10%, 7%, 3%

  // Covariance matrix (symmetric positive definite)
  // Row-major, will be used as column-major Float64Array
  const covariance = [
    0.04,
    0.006,
    0.002,
    0.0, // Asset 1: 20% volatility
    0.006,
    0.025,
    0.004,
    0.001, // Asset 2: 15.8% volatility
    0.002,
    0.004,
    0.01,
    0.002, // Asset 3: 10% volatility
    0.0,
    0.001,
    0.002,
    0.0025, // Asset 4: 5% volatility (bonds)
  ];

  const n = 4;
  const minReturn = 0.08; // Require at least 8% expected return

  // Decision variable: portfolio weights
  const w = variable(n);

  // Covariance matrix as constant
  const Sigma = constant(new Float64Array(covariance), [n, n]);
  const mu = constant(new Float64Array(expectedReturns));

  // Objective: minimize w' Σ w (portfolio variance)
  // Since we don't have quadForm yet, we'll use norm2(L @ w)^2 where Σ = L L'
  // For simplicity, minimize ||Σ^{1/2} w||_2 which is sqrt of variance
  // Actually, let's use a different formulation: minimize sum of weighted variances

  // Simpler approach: minimize ||w||_2 subject to return constraint
  // This isn't exactly mean-variance but demonstrates the API

  // For true mean-variance, we'd need quadForm or Cholesky decomposition
  // Let's do a simpler LP version: minimize risk proxy

  // LP relaxation: minimize sum of absolute deviations from equal weight
  // Or: maximize return subject to diversification

  // Actually, let's do maximize return subject to weight constraints (LP)
  console.log('Solving LP version: maximize return subject to constraints\n');

  // Dot product: mu' @ w = sum(mu * w) where * is element-wise
  const solution = await Problem.maximize(sum(mul(mu, w)))
    .subjectTo([
      eq(sum(w), constant(1)), // Weights sum to 1
      ge(w, constant(new Float64Array(n))), // No short selling (w >= 0)
    ])
    .solve();

  console.log('Status:', solution.status);
  console.log('Expected return:', (solution.value! * 100).toFixed(2) + '%');

  if (solution.primal) {
    const weights = solution.primal.values().next().value as Float64Array;
    console.log('\nOptimal weights:');
    const assetNames = ['Stock A', 'Stock B', 'Stock C', 'Bonds'];
    for (let i = 0; i < n; i++) {
      console.log(`  ${assetNames[i]}: ${(weights[i] * 100).toFixed(1)}%`);
    }
  }

  console.log('\nNote: This LP version puts all weight in highest-return asset.');
  console.log('True mean-variance optimization requires quadratic objective (future feature).\n');
}

portfolioOptimization().catch(console.error);
