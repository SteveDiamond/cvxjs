/**
 * Least Squares Regression Example
 *
 * Minimize the 2-norm of residuals (linear regression).
 *
 * Problem:
 *   minimize    ||A x - b||_2
 *
 * where:
 *   A = design matrix (m x n)
 *   b = observations (m)
 *   x = parameters to fit (n)
 */

import { variable, constant, Problem } from '../src/index.js';

async function leastSquares() {
  console.log('=== Least Squares Regression ===\n');

  // Simple 2D linear fit: y = a + b*x
  const points = [
    [1, 2.1],
    [2, 3.9],
    [3, 6.2],
    [4, 7.8],
    [5, 10.1],
  ];

  const m = points.length;

  // Design matrix: [1, x] for each point
  const A = constant(points.map(([xi]) => [1, xi]));
  const b = constant(points.map(([, yi]) => yi));

  // Decision variable: [intercept, slope]
  const x = variable(2);

  // Objective: minimize ||A @ x - b||_2
  const residual = A.matmul(x).sub(b);

  console.log('Fitting line y = a + b*x to data points:');
  for (const [xi, yi] of points) {
    console.log(`  (${xi}, ${yi})`);
  }
  console.log();

  const solution = await Problem.minimize(residual.norm2()).solve();

  console.log('Status:', solution.status);
  console.log('Residual norm:', solution.value?.toFixed(4));

  const params = solution.valueOf(x)!;
  const [intercept, slope] = params;

  console.log(`\nFitted line: y = ${intercept.toFixed(3)} + ${slope.toFixed(3)} * x`);

  // Calculate R²
  const yMean = points.reduce((sum, [, yi]) => sum + yi, 0) / m;
  let ssTot = 0;
  let ssRes = 0;
  for (const [xi, yi] of points) {
    const yPred = intercept + slope * xi;
    ssTot += (yi - yMean) ** 2;
    ssRes += (yi - yPred) ** 2;
  }
  const r2 = 1 - ssRes / ssTot;

  console.log(`R² = ${r2.toFixed(4)}`);
  console.log('\nPredictions:');
  for (const [xi, yi] of points) {
    const yPred = intercept + slope * xi;
    console.log(`  x=${xi}: predicted=${yPred.toFixed(2)}, actual=${yi}`);
  }

  console.log();
}

leastSquares().catch(console.error);
