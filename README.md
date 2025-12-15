# cvxjs

Disciplined Convex Programming in TypeScript. A CVXPY-inspired library for convex optimization that runs in both Node.js and browsers.

## Features

- **DCP Verification**: Automatic validation of convex programs using disciplined convex programming rules
- **Rich Expression System**: Variables, constants, arithmetic operations, and convex atoms
- **Multiple Problem Types**: LP, SOCP support via Clarabel solver
- **Universal**: Works in Node.js and browsers via WebAssembly
- **TypeScript-First**: Full type safety with builder pattern API

## Installation

```bash
npm install cvxjs
```

## Quick Start

```typescript
import { variable, constant, sum, norm2, ge, eq, Problem } from 'cvxjs';

// Linear Program: minimize sum(x) subject to x >= 1
const x = variable(3);
const solution = await Problem.minimize(sum(x))
  .subjectTo([ge(x, constant([1, 1, 1]))])
  .solve();

console.log('Optimal value:', solution.value);  // 3
console.log('Status:', solution.status);        // 'optimal'
```

## Examples

### Linear Programming

```typescript
import { variable, constant, sum, ge, le, Problem } from 'cvxjs';

// Bounded LP: maximize sum(x) subject to 0 <= x <= 2
const x = variable(3);
const solution = await Problem.maximize(sum(x))
  .subjectTo([
    ge(x, constant([0, 0, 0])),
    le(x, constant([2, 2, 2])),
  ])
  .solve();

console.log('Optimal value:', solution.value);  // 6
```

### Second-Order Cone Programming

```typescript
import { variable, constant, norm2, sum, eq, Problem } from 'cvxjs';

// Minimize ||x||_2 subject to sum(x) = 3
const x = variable(3);
const solution = await Problem.minimize(norm2(x))
  .subjectTo([eq(sum(x), constant(3))])
  .solve();

console.log('||x||_2 =', solution.value);  // sqrt(3) ≈ 1.732
```

### L1 Norm Minimization (Lasso-style)

```typescript
import { variable, constant, sub, matmul, norm2, norm1, add, mul, Problem } from 'cvxjs';

// Lasso: minimize ||X @ beta - y||_2 + lambda * ||beta||_1
const X = constant([[1, 2], [3, 4], [5, 6]]);  // 3x2 matrix
const y = constant([1, 2, 3]);
const beta = variable(2);
const lambda = 0.1;

const residual = sub(matmul(X, beta), y);
const objective = add(norm2(residual), mul(constant(lambda), norm1(beta)));

const solution = await Problem.minimize(objective).solve();
```

### Least Squares Regression

```typescript
import { variable, constant, sub, matmul, norm2, Problem } from 'cvxjs';

// Fit y = a + b*x
const points = [[1, 2.1], [2, 3.9], [3, 6.2], [4, 7.8], [5, 10.1]];
const A = constant(points.map(([x]) => [1, x]));  // Design matrix
const b = constant(points.map(([, y]) => y));

const params = variable(2);  // [intercept, slope]
const solution = await Problem.minimize(norm2(sub(matmul(A, params), b))).solve();
```

## API Reference

### Variables and Constants

```typescript
variable(n)                    // n-dimensional variable
constant(5)                    // Scalar constant
constant([1, 2, 3])           // Vector constant
constant([[1, 2], [3, 4]])    // Matrix constant
zeros(n), ones(n)             // Zero/ones vectors
zeros(m, n), ones(m, n)       // Zero/ones matrices
eye(n)                        // Identity matrix
```

### Arithmetic Operations

```typescript
add(x, y)      // x + y
sub(x, y)      // x - y
mul(x, y)      // Element-wise x * y (one must be constant)
div(x, c)      // x / c (c must be constant scalar)
neg(x)         // -x
matmul(A, x)   // Matrix multiplication A @ x
```

### Convex Atoms

```typescript
sum(x)         // Sum of elements
norm1(x)       // L1 norm: sum(|x_i|)
norm2(x)       // L2 norm: sqrt(sum(x_i^2))
normInf(x)     // Infinity norm: max(|x_i|)
abs(x)         // Element-wise absolute value
pos(x)         // Element-wise max(x, 0)
maximum(...)   // Element-wise maximum
minimum(...)   // Element-wise minimum (concave)
```

### Constraints

```typescript
eq(x, y)       // Equality: x == y
ge(x, y)       // Greater-equal: x >= y
le(x, y)       // Less-equal: x <= y
```

### Problem Builder

```typescript
Problem.minimize(objective)    // Create minimization problem
Problem.maximize(objective)    // Create maximization problem
  .subjectTo([...constraints]) // Add constraints
  .settings({ verbose: true }) // Configure solver
  .solve()                     // Solve and return Promise<Solution>
```

### Solution Object

```typescript
interface Solution {
  status: 'optimal' | 'infeasible' | 'unbounded' | 'max_iterations' | 'numerical_error';
  value?: number;              // Optimal objective value
  primal?: Map<ExprId, Float64Array>;  // Variable values
  solveTime?: number;          // Solve time in seconds
  iterations?: number;         // Solver iterations
}
```

## Supported Problem Types

| Type | Description | Atoms |
|------|-------------|-------|
| **LP** | Linear Programming | `sum`, `add`, `sub`, `mul` |
| **SOCP** | Second-Order Cone | `norm2`, `norm1`, `normInf`, `abs` |

## DCP Rules

cvxjs enforces Disciplined Convex Programming rules:

- **Minimization**: Objective must be convex
- **Maximization**: Objective must be concave
- **Equality constraints**: Must be affine
- **Inequality constraints**: `expr >= 0` where `expr` is concave, or `expr <= 0` where `expr` is convex

The library will throw a `DcpError` if your problem violates these rules.

## Browser Usage

cvxjs works in browsers via WebAssembly:

```html
<script type="module">
import { variable, constant, sum, ge, Problem } from 'cvxjs';

const x = variable(3);
const solution = await Problem.minimize(sum(x))
  .subjectTo([ge(x, constant([1, 1, 1]))])
  .solve();

console.log('Solution:', solution.value);
</script>
```

## Architecture

- **Expression System**: Immutable expression trees with shape inference
- **DCP Analysis**: Curvature and sign propagation for convexity verification
- **Canonicalization**: Transform expressions to standard conic form
- **Solver**: Clarabel via WebAssembly (Rust compiled to WASM)

## Running Examples

```bash
npm run example:basic          # Basic API usage
npm run example:least-squares  # Linear regression
npm run example:lasso          # L1-regularized regression
npm run example:portfolio      # Portfolio optimization
```

## Development

```bash
npm install          # Install dependencies
npm run build        # Build the library
npm run test         # Run tests in watch mode
npm run test:run     # Run tests once
npm run build:wasm   # Rebuild WASM (requires Rust + wasm-pack)
```

## License

Apache-2.0

## Acknowledgments

- [CVXPY](https://www.cvxpy.org/) - The inspiration for this library
- [Clarabel](https://github.com/oxfordcontrol/Clarabel.rs) - The underlying conic solver
