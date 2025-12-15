/**
 * Lasso Regression Example
 *
 * L1-regularized least squares for sparse linear regression.
 *
 * Problem:
 *   minimize    ||X β - y||_2^2 + λ ||β||_1
 *
 * Reformulated for SOCP:
 *   minimize    t + λ ||β||_1
 *   subject to  ||X β - y||_2 <= t
 *
 * where:
 *   X = feature matrix (m x n)
 *   y = target vector (m)
 *   β = coefficients (n)
 *   λ = regularization parameter
 */

import {
  variable,
  constant,
  add,
  sub,
  mul,
  matmul,
  norm1,
  norm2,
  le,
  Problem,
} from '../src/index.js';

async function lassoRegression() {
  console.log('=== Lasso Regression ===\n');

  // Generate synthetic data: y = X @ beta_true + noise
  // True coefficients are sparse: only a few are non-zero
  const m = 20; // samples
  const n = 5; // features

  // True sparse coefficients
  const betaTrue = [3.0, 0.0, 0.0, -2.0, 0.0];

  // Feature matrix (random-ish values for reproducibility)
  // Build as 2D array for proper shape inference
  const X_rows: number[][] = [];
  for (let i = 0; i < m; i++) {
    const row: number[] = [];
    for (let j = 0; j < n; j++) {
      // Simple deterministic "random" pattern
      row.push(Math.sin(i * 0.5 + j * 1.3) + Math.cos(i * 0.3 - j * 0.7));
    }
    X_rows.push(row);
  }

  // Generate y = X @ beta_true (+ small noise would be added in practice)
  const y_data: number[] = [];
  for (let i = 0; i < m; i++) {
    let sum = 0;
    for (let j = 0; j < n; j++) {
      sum += X_rows[i][j] * betaTrue[j];
    }
    y_data.push(sum + 0.1 * Math.sin(i)); // small noise
  }

  const X = constant(X_rows); // m x n matrix
  const y = constant(y_data); // m vector

  // Decision variables
  const beta = variable(n);

  // Regularization parameter
  const lambda = 0.5;

  // Residual: X @ beta - y
  const residual = sub(matmul(X, beta), y);

  // Lasso objective: ||residual||_2 + lambda * ||beta||_1
  // Note: We minimize ||residual||_2 (not squared) + lambda * ||beta||_1
  // This is slightly different from standard Lasso but is a valid SOCP
  const objective = add(norm2(residual), mul(constant(lambda), norm1(beta)));

  console.log(`Data: ${m} samples, ${n} features`);
  console.log(`True coefficients: [${betaTrue.join(', ')}]`);
  console.log(`Regularization λ = ${lambda}\n`);

  const solution = await Problem.minimize(objective).solve();

  console.log('Status:', solution.status);
  console.log('Objective value:', solution.value?.toFixed(4));

  if (solution.primal) {
    const betaOpt = solution.primal.values().next().value as Float64Array;
    console.log('\nRecovered coefficients:');
    for (let j = 0; j < n; j++) {
      const recovered = betaOpt[j].toFixed(3);
      const truth = betaTrue[j].toFixed(3);
      const marker = Math.abs(betaOpt[j]) < 0.1 ? ' (sparse)' : '';
      console.log(`  β[${j}] = ${recovered.padStart(7)} (true: ${truth})${marker}`);
    }
  }

  console.log('\nNote: L1 regularization encourages sparse solutions.');
  console.log('Coefficients close to zero demonstrate the sparsity-inducing property.\n');
}

lassoRegression().catch(console.error);
