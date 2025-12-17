# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
npm run build          # Build library with tsup
npm run test           # Run tests in watch mode (vitest)
npm run test:run       # Run tests once
npm run lint           # ESLint on src and tests
npm run typecheck      # TypeScript type checking
npm run format:check   # Check Prettier formatting

# Run single example
npm run example:basic
npm run example:lasso
npm run example:least-squares
npm run example:portfolio

# WASM rebuild (requires Rust + wasm-pack)
npm run build:wasm
```

## Architecture

cvxjs is a CVXPY-inspired convex optimization library for TypeScript/JavaScript.

### Core Pipeline

1. **Expression System** (`src/expr/`) - Immutable expression trees representing optimization variables, constants, and operations. Each expression has a shape and carries DCP metadata.

2. **DCP Analysis** (`src/dcp/`) - Curvature and sign propagation to verify convexity. Expressions are classified as convex/concave/affine/constant.

3. **Canonicalization** (`src/canon/`) - Transforms DCP-compliant expressions into standard conic form:
   - `lin-expr.ts` - Linear expression representation
   - `quad-expr.ts` - Quadratic expression representation (for QP)
   - `canonicalizer.ts` - Expression-to-conic transformation
   - `stuffing.ts` - Builds final sparse matrices for solver

4. **Solver Backend** (`src/solver/`) - Two solvers available:
   - Clarabel (WASM) - Conic optimization (LP, QP, SOCP)
   - HiGHS - Linear/Mixed-Integer Programming
   - `router.ts` - Selects solver based on problem properties

### Key Types

- `ExprData` - Internal expression representation (union type with `kind` discriminator)
- `Expr` - User-facing wrapper with fluent API methods
- `Constraint` - Zero/Nonneg/SOC constraint types
- `ConeConstraint` - Canonicalized cone constraint
- `LinExpr` / `QuadExpr` - Canonical linear/quadratic expressions
- `CscMatrix` - Compressed Sparse Column matrix format

### Expression Kinds

Expressions use discriminated unions (`kind` field):
- `variable`, `constant` - Leaf nodes
- `add`, `sub`, `mul`, `div`, `matmul` - Binary ops
- `neg`, `sum`, `norm1`, `norm2`, `normInf`, `abs`, etc. - Unary atoms
- `quadForm`, `quadOverLin` - Quadratic atoms

### Solver Selection

Router (`src/solver/router.ts`) automatically selects:
- **HiGHS**: Integer/binary variables present
- **Clarabel**: All other problems (LP, QP, SOCP)

Mixed-integer conic problems are not supported (throws `DcpError`).

## Code Style

- Imports at the top of files
- All exports from `src/index.ts` are the public API
- Tests in `tests/unit/` (unit) and `tests/integration/` (solver tests)
- Sparse matrices use CSC format throughout
