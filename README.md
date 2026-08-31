# fixed-dim-qp

A fixed-dimension-3, allocation-free, randomized-incremental (Seidel/SDLP-style)
solver for the Euclidean projection of a point onto an intersection of
halfspaces:

```
p* = argmin_{p} ||p - q||^2   s.t.   A p >= b
```

This is the primitive behind the regularized progressive-hulls placement QP

```
min_p  g^T p + (lambda/2) ||p - p0||^2   s.t.   A p >= b
```

(see `progressive_hulls_fixed_dim_qp_design.md` for the full design spec and
the reduction from the regularized objective to a plain projection via
`q = p0 - g/(2*lambda)`).

## Status

The solver, exhaustive oracle, and sections A-C of the design spec's test
plan (deterministic, randomized differential, and metamorphic tests) are
implemented and passing in CI. Section D/E (mesh-level regression, stress/
fuzz, performance) live in the separate bench repo below, along with the
accuracy/perf comparison against `igl::copyleft::quadprog` and
`igl::linprog`.

## Usage

```cpp
#include <igl/project_to_halfspace_intersection.h>

Eigen::Vector3d q(5,5,5); // preferred point
Eigen::Matrix<double,Eigen::Dynamic,3> A(3,3); // row i: halfspace normal a_i
A << 1,0,0,  0,1,0,  0,0,1;
Eigen::VectorXd b(3); b << 0,0,0; // row i: halfspace offset b_i (a_i^T p >= b_i)

igl::HalfspaceProjectionOptions<double> options;
igl::HalfspaceProjectionResult<double,3> result;
const auto status = igl::project_to_halfspace_intersection<double,3>(q, A, b, options, result);
// status == igl::HalfspaceProjectionStatus::SUCCESS
// result.p == (1,1,1), result.active_count == 3
```

To solve the regularized progressive-hulls placement QP
`min_p g^T p + (lambda/2)||p-p0||^2 s.t. Ap>=b`, reduce it to a projection
first: `q = p0 - g/(2*lambda)`.

## Layout

This repo is laid out to match libigl's `include/igl/` module convention
exactly, so the header(s) here are intended to be droppable, unmodified, into
a libigl checkout:

```
include/igl/project_to_halfspace_intersection.h
include/igl/project_to_halfspace_intersection.cpp
```

`tests/` holds the exhaustive oracle, deterministic unit tests, randomized
differential tests against the oracle, and metamorphic tests, per the design
spec's test plan (sections A-C; sections D-E, which need real meshes, live in
the separate [`fixed-dim-qp-bench`](https://github.com/alecjacobson/fixed-dim-qp-bench)
repo).

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
ctest --test-dir build --output-on-failure
```

## License

MPL-2.0, matching libigl, so this can be merged upstream without a license
change.
