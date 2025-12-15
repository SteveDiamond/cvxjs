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

import {
  variable,
  constant,
  sub,
  matmul,
  norm2,
  Problem,
} from '../src/index.js';

async function leastSquares() {
  console.log('=== Least Squares Regression ===\n');

  // Simple 2D linear fit: y = a + b*x
  // We'll fit a line to some data points

  const points = [
    [1, 2.1],
    [2, 3.9],
    [3, 6.2],
    [4, 7.8],
    [5, 10.1],
  ];

  const m = points.length; // number of data points
  const n = 2; // parameters: [intercept, slope]

  // Design matrix: [1, x] for each point (as 2D array for proper shape inference)
  const A_rows: number[][] = [];
  const b_data: number[] = [];

  for (let i = 0; i < m; i++) {
    A_rows.push([1, points[i][0]]); // [intercept, x_value]
    b_data.push(points[i][1]); // y values
  }

  const A = constant(A_rows); // 5x2 matrix
  const b = constant(b_data); // 5 vector

  // Decision variable: [intercept, slope]
  const x = variable(n);

  // Objective: minimize ||A x - b||_2
  const residual = sub(matmul(A, x), b);
  const objective = norm2(residual);

  console.log('Fitting line y = a + b*x to data points:');
  for (const [xi, yi] of points) {
    console.log(`  (${xi}, ${yi})`);
  }
  console.log();

  const solution = await Problem.minimize(objective).solve();

  console.log('Status:', solution.status);
  console.log('Residual norm:', solution.value?.toFixed(4));

  if (solution.primal) {
    const params = solution.primal.values().next().value as Float64Array;
    const intercept = params[0];
    const slope = params[1];

    console.log(`\nFitted line: y = ${intercept.toFixed(3)} + ${slope.toFixed(3)} * x`);

    // Calculate R² (coefficient of determination)
    const yMean = b_data.reduce((a, b) => a + b, 0) / m;
    let ssTot = 0;
    let ssRes = 0;
    for (let i = 0; i < m; i++) {
      const yPred = intercept + slope * points[i][0];
      ssTot += (points[i][1] - yMean) ** 2;
      ssRes += (points[i][1] - yPred) ** 2;
    }
    const r2 = 1 - ssRes / ssTot;

    console.log(`R² = ${r2.toFixed(4)}`);
    console.log('\nPredictions:');
    for (const [xi, yi] of points) {
      const yPred = intercept + slope * xi;
      console.log(`  x=${xi}: predicted=${yPred.toFixed(2)}, actual=${yi}`);
    }
  }

  console.log();
}

leastSquares().catch(console.error);
