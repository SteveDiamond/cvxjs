# Changelog

All notable changes to this project will be documented in this file.

## [0.1.1] - 2025-12-16

### Added

- **Fluent API**: Method chaining on expressions for cleaner code
  - Arithmetic: `x.add(y)`, `x.sub(y)`, `x.mul(c)`, `x.div(c)`, `x.neg()`, `A.matmul(x)`, `x.dot(y)`
  - Atoms: `x.sum()`, `x.norm1()`, `x.norm2()`, `x.normInf()`, `x.abs()`, `x.pos()`
  - Constraints: `x.eq(y)`, `x.ge(y)`, `x.le(y)`
- **Solution.valueOf(x)**: Convenient method to get variable values from solution
- **Expr.value(primal)**: Evaluate any expression at a solution point
- **Array auto-wrapping**: Pass raw arrays directly to constraint methods (`x.le([1,2,3])`)
- **Scalar broadcasting**: Use scalars with any-size expressions (`x.ge(0)`)
- **QP support**: `sumSquares(x)` and `quadForm(x, P)` for quadratic programming

### Changed

- Updated README with fluent API examples (significantly improved readability)
- Added QP to supported problem types documentation

## [0.1.0] - 2025-12-15

### Added

- Initial release
- Disciplined Convex Programming (DCP) verification
- Expression system with shape inference
- LP, SOCP support via Clarabel solver (WebAssembly)
- Works in Node.js and browsers
- Core atoms: `sum`, `norm1`, `norm2`, `normInf`, `abs`, `pos`, `maximum`, `minimum`
- Constraint types: equality, inequality
- Problem builder API with `minimize`/`maximize`
