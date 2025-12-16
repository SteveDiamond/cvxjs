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

import { variable, constant, Problem } from '../src/index.js';

async function portfolioOptimization() {
  console.log('=== Portfolio Optimization (Mean-Variance) ===\n');

  // 4 assets with expected returns and covariance matrix
  const expectedReturns = [0.12, 0.1, 0.07, 0.03]; // 12%, 10%, 7%, 3%

  // Covariance matrix (symmetric positive definite)
  // Asset volatilities: 20%, 15.8%, 10%, 5%
  const covariance = [
    [0.04, 0.006, 0.002, 0.0],
    [0.006, 0.025, 0.004, 0.001],
    [0.002, 0.004, 0.01, 0.002],
    [0.0, 0.001, 0.002, 0.0025],
  ];

  const n = 4;
  const minReturn = 0.08;
  const riskAversion = 2.0;
  const assetNames = ['Stock A (12%)', 'Stock B (10%)', 'Stock C (7%)', 'Bonds (3%)'];

  // Decision variable: portfolio weights
  const w = variable(n);

  // Constants
  const Sigma = constant(covariance);
  const mu = constant(expectedReturns);

  // Expressions for analysis
  const portfolioVariance = w.quadForm(Sigma);
  const portfolioReturn = mu.mul(w).sum();

  // =====================================================
  // Example 1: Minimize variance subject to return constraint
  // =====================================================
  console.log('--- Example 1: Minimum Variance Portfolio ---');
  console.log(`minimize w'Σw subject to μ'w >= ${minReturn * 100}%, sum(w) = 1, w >= 0\n`);

  {
    const solution = await Problem.minimize(portfolioVariance)
      .subjectTo([
        portfolioReturn.ge(minReturn),
        w.sum().eq(1),
        w.ge(0),
      ])
      .solve();

    console.log('Status:', solution.status);
    console.log('Portfolio variance:', solution.value?.toFixed(6));
    console.log(
      'Portfolio std dev:',
      Math.sqrt(solution.value!).toFixed(4),
      '(' + (Math.sqrt(solution.value!) * 100).toFixed(2) + '%)'
    );

    const weights = solution.valueOf(w)!;
    console.log('\nOptimal weights:');
    for (let i = 0; i < n; i++) {
      console.log(`  ${assetNames[i]}: ${(weights[i] * 100).toFixed(1)}%`);
    }

    // Use expression evaluation instead of manual calculation
    const expReturn = portfolioReturn.value(solution.primal!);
    console.log(`\nExpected return: ${(expReturn * 100).toFixed(2)}%`);
    console.log();
  }

  // =====================================================
  // Example 2: Mean-Variance Tradeoff (Risk Aversion)
  // =====================================================
  console.log('--- Example 2: Mean-Variance Tradeoff ---');
  console.log(`minimize (${riskAversion} * w'Σw - μ'w) subject to sum(w) = 1, w >= 0\n`);

  {
    // Objective: minimize λ * variance - return
    const objective = portfolioVariance.mul(riskAversion).sub(portfolioReturn);

    const solution = await Problem.minimize(objective)
      .subjectTo([
        w.sum().eq(1),
        w.ge(0),
      ])
      .solve();

    console.log('Status:', solution.status);

    const weights = solution.valueOf(w)!;
    console.log('\nOptimal weights:');
    for (let i = 0; i < n; i++) {
      console.log(`  ${assetNames[i]}: ${(weights[i] * 100).toFixed(1)}%`);
    }

    // Use expression evaluation - no manual recalculation needed!
    const expReturn = portfolioReturn.value(solution.primal!);
    const variance = portfolioVariance.value(solution.primal!);

    console.log(`\nExpected return: ${(expReturn * 100).toFixed(2)}%`);
    console.log(`Portfolio std dev: ${(Math.sqrt(variance) * 100).toFixed(2)}%`);
    console.log();
  }

  console.log('=== Portfolio optimization complete ===\n');
}

portfolioOptimization().catch(console.error);
