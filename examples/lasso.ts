/**
 * Lasso Regression Example
 *
 * L1-regularized least squares for sparse linear regression.
 *
 * Problem:
 *   minimize    ||X β - y||_2 + λ ||β||_1
 *
 * where:
 *   X = feature matrix (m x n)
 *   y = target vector (m)
 *   β = coefficients (n)
 *   λ = regularization parameter
 */

import { variable, constant, Problem } from '../src/index.js';

async function lassoRegression() {
  console.log('=== Lasso Regression ===\n');

  // Generate synthetic data: y = X @ beta_true + noise
  // True coefficients are sparse: only a few are non-zero
  const m = 20; // samples
  const n = 5; // features

  // True sparse coefficients
  const betaTrue = [3.0, 0.0, 0.0, -2.0, 0.0];

  // Feature matrix (deterministic pattern for reproducibility)
  const X_data: number[][] = [];
  for (let i = 0; i < m; i++) {
    const row: number[] = [];
    for (let j = 0; j < n; j++) {
      row.push(Math.sin(i * 0.5 + j * 1.3) + Math.cos(i * 0.3 - j * 0.7));
    }
    X_data.push(row);
  }

  // Generate y = X @ beta_true + small noise
  const y_data: number[] = [];
  for (let i = 0; i < m; i++) {
    let val = 0;
    for (let j = 0; j < n; j++) {
      val += X_data[i][j] * betaTrue[j];
    }
    y_data.push(val + 0.1 * Math.sin(i));
  }

  const X = constant(X_data);
  const y = constant(y_data);

  // Decision variable
  const beta = variable(n);

  // Regularization parameter
  const lambda = 0.5;

  // Residual: X @ beta - y
  const residual = X.matmul(beta).sub(y);

  // Lasso objective: ||residual||_2 + λ * ||β||_1
  const objective = residual.norm2().add(beta.norm1().mul(lambda));

  console.log(`Data: ${m} samples, ${n} features`);
  console.log(`True coefficients: [${betaTrue.join(', ')}]`);
  console.log(`Regularization λ = ${lambda}\n`);

  const solution = await Problem.minimize(objective).solve();

  console.log('Status:', solution.status);
  console.log('Objective value:', solution.value?.toFixed(4));

  const betaOpt = solution.valueOf(beta)!;
  console.log('\nRecovered coefficients:');
  for (let j = 0; j < n; j++) {
    const recovered = betaOpt[j].toFixed(3);
    const truth = betaTrue[j].toFixed(3);
    const marker = Math.abs(betaOpt[j]) < 0.1 ? ' (sparse)' : '';
    console.log(`  β[${j}] = ${recovered.padStart(7)} (true: ${truth})${marker}`);
  }

  console.log('\nNote: L1 regularization encourages sparse solutions.');
  console.log('Coefficients close to zero demonstrate the sparsity-inducing property.\n');
}

lassoRegression().catch(console.error);
