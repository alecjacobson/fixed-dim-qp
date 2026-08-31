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

Under active development. See the design spec for the full contract, test
plan, and acceptance criteria.

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
