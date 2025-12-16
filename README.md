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
import { variable, Problem } from 'cvxjs';

// Linear Program: minimize sum(x) subject to x >= 1
const x = variable(3);
const solution = await Problem.minimize(x.sum())
  .subjectTo([x.ge(1)])
  .solve();

console.log('Optimal value:', solution.value);    // 3
console.log('Status:', solution.status);          // 'optimal'
console.log('x =', solution.valueOf(x));          // Float64Array [1, 1, 1]
```

## Examples

### Linear Programming

```typescript
import { variable, Problem } from 'cvxjs';

// Bounded LP: maximize sum(x) subject to 0 <= x <= 2
const x = variable(3);
const solution = await Problem.maximize(x.sum())
  .subjectTo([x.ge(0), x.le(2)])
  .solve();

console.log('Optimal value:', solution.value);  // 6
console.log('x =', solution.valueOf(x));        // Float64Array [2, 2, 2]
```

### Second-Order Cone Programming

```typescript
import { variable, Problem } from 'cvxjs';

// Minimize ||x||_2 subject to sum(x) = 3
const x = variable(3);
const solution = await Problem.minimize(x.norm2())
  .subjectTo([x.sum().eq(3)])
  .solve();

console.log('||x||_2 =', solution.value);  // sqrt(3) ≈ 1.732
```

### L1 Norm Minimization (Lasso-style)

```typescript
import { variable, constant, Problem } from 'cvxjs';

// Lasso: minimize ||X @ beta - y||_2 + lambda * ||beta||_1
const X = constant([[1, 2], [3, 4], [5, 6]]);  // 3x2 matrix
const y = constant([1, 2, 3]);
const beta = variable(2);
const lambda = 0.1;

const residual = X.matmul(beta).sub(y);
const objective = residual.norm2().add(beta.norm1().mul(lambda));

const solution = await Problem.minimize(objective).solve();
```

### Least Squares Regression

```typescript
import { variable, constant, Problem } from 'cvxjs';

// Fit y = a + b*x
const points = [[1, 2.1], [2, 3.9], [3, 6.2], [4, 7.8], [5, 10.1]];
const A = constant(points.map(([x]) => [1, x]));  // Design matrix
const b = constant(points.map(([, y]) => y));

const params = variable(2);  // [intercept, slope]
const residual = A.matmul(params).sub(b);
const solution = await Problem.minimize(residual.norm2()).solve();

console.log('intercept, slope:', solution.valueOf(params));
```

### Quadratic Programming

```typescript
import { variable, constant, quadForm, Problem } from 'cvxjs';

// Minimize 0.5 * x'Px + q'x subject to x >= 0
const P = constant([[2, 0], [0, 2]]);  // Positive definite matrix
const q = constant([-2, -5]);
const x = variable(2);

const objective = quadForm(x, P).mul(0.5).add(q.dot(x));
const solution = await Problem.minimize(objective)
  .subjectTo([x.ge(0)])
  .solve();

console.log('x =', solution.valueOf(x));  // [1, 2.5]
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

Expressions support fluent method chaining. Arrays and scalars are auto-wrapped.

```typescript
// Fluent API (recommended)
x.add(y)       // x + y
x.sub(y)       // x - y
x.mul(0.5)     // Element-wise x * 0.5
x.div(2)       // x / 2
x.neg()        // -x
A.matmul(x)    // Matrix multiplication A @ x
x.dot(y)       // Dot product

// Function API (also available)
add(x, y), sub(x, y), mul(x, y), div(x, c), neg(x), matmul(A, x)
```

### Convex Atoms

```typescript
// Fluent API
x.sum()        // Sum of elements
x.norm1()      // L1 norm: sum(|x_i|)
x.norm2()      // L2 norm: sqrt(sum(x_i^2))
x.normInf()    // Infinity norm: max(|x_i|)
x.abs()        // Element-wise absolute value
x.pos()        // Element-wise max(x, 0)
x.maximum(y)   // Element-wise maximum
x.minimum(y)   // Element-wise minimum (concave)

// Quadratic (for QP)
sumSquares(x)     // ||x||_2^2 = sum(x_i^2)
quadForm(x, P)    // x'Px (P must be positive semidefinite)

// Function API also available
sum(x), norm1(x), norm2(x), normInf(x), abs(x), pos(x)
```

### Constraints

Constraints support scalar broadcasting and array auto-wrapping.

```typescript
// Fluent API (recommended)
x.eq(y)        // Equality: x == y
x.ge(0)        // Greater-equal: x >= 0 (scalar broadcasts)
x.le([1,2,3])  // Less-equal: x <= [1,2,3] (array auto-wrapped)

// Function API
eq(x, y), ge(x, y), le(x, y)
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
  solveTime?: number;          // Solve time in seconds
  iterations?: number;         // Solver iterations

  valueOf(x: Expr): Float64Array | undefined;  // Get variable values
}

// Get variable values at solution
const xValues = solution.valueOf(x);  // Float64Array

// Evaluate any expression at solution
const residualValue = residual.value(solution.primal);  // number | Float64Array
```

## Supported Problem Types

| Type | Description | Atoms |
|------|-------------|-------|
| **LP** | Linear Programming | `sum`, `add`, `sub`, `mul` |
| **QP** | Quadratic Programming | `sumSquares`, `quadForm` |
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
import { variable, Problem } from 'cvxjs';

const x = variable(3);
const solution = await Problem.minimize(x.sum())
  .subjectTo([x.ge(1)])
  .solve();

console.log('Solution:', solution.value);
console.log('x =', solution.valueOf(x));
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
