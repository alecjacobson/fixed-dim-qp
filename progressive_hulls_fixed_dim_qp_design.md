# Design spec: fixed-dimensional QP placement for progressive hulls

## Purpose

Replace the placement solve in a progressive-hulls edge collapse with a small,
robust, allocation-free convex quadratic-programming solver. The solver must
preserve the half-space containment constraints while retaining the quadratic
regularization used by the libigl implementation.

This is a development and test specification. It deliberately does not prescribe
repository layout, public API naming, or the mesh-collapse implementation beyond
the data needed by the placement solver.

## Problem

For each candidate edge collapse, construct a set of oriented halfspaces

\[
  a_i^T p \ge b_i, \qquad i=1,\ldots,m,
\]

such that a feasible merged position \(p\in\mathbb R^3\) moves every affected
face outward. Feasibility is the geometric containment invariant; the objective
only chooses a preferred feasible point.

The original Progressive Hulls objective is linear because local volume change
is linear in the merged-vertex position:

\[
  \min_p\; g^T p \quad\text{subject to}\quad Ap\ge b.
\]

The target formulation is the strictly convex regularized problem

\[
  \min_p\; g^T p + \frac12(p-p_0)^TQ(p-p_0)
  \quad\text{subject to}\quad Ap\ge b,
\]

where \(Q\succ0\). The usual and preferred first implementation has
\(Q=\lambda I\), \(\lambda>0\), and `p0` equal to the edge midpoint or another
local reference position.

Do **not** replace this by an LP followed by a tie break. At finite
regularization weight, the QP can intentionally accept a small volume increase
for a much better-centered position. That behavior is part of the placement
model, not an implementation detail.

## Mathematical reduction

For isotropic regularization,

\[
  g^Tp + \lambda\|p-p_0\|^2
  = \lambda\|p-q\|^2+\mathrm{constant},
  \qquad q=p_0-\frac{g}{2\lambda}.
\]

Thus solve the Euclidean projection problem

\[
  p^\star=\operatorname*{argmin}_{Ap\ge b}\|p-q\|^2.
\]

For a dense SPD `Q`, factor `Q = R^T R`, set `y = R p`, transform each
halfspace to `a_i^T R^{-1} y >= b_i`, and perform the same Euclidean projection
in `y` coordinates. Support the isotropic case first; add the whitening path
only when a real caller needs anisotropic regularization.

## Non-goals

- A general LP/QP library.
- Sparse matrices, dynamic allocations, or arbitrary-dimensional optimization.
- Replacing the separate mesh validity checks: link condition, inversion,
  self-intersection, and priority-queue management remain outside this solver.
- Claiming exact containment from floating-point arithmetic alone.

## Solver choice

Implement a specialized **randomized incremental closest-point-to-polyhedron
solver** in dimension 3. It is a Seidel/SDLP-style recursion, but its objective
is a quadratic projection rather than a linear objective.

Why this is the primary implementation:

- Dimension is fixed at three; a solution has at most three linearly independent
  support planes.
- The local number of planes is a vertex-ring quantity, not a global problem.
- It uses only dot products and tiny orthonormal-basis calculations.
- Expected work is linear in the number of constraints for fixed dimension.
- Strict convexity gives one placement, avoiding LP degeneracy and arbitrary
  vertex choices.

The existing fixed-dimensional LP implementation is a useful structural model,
but it cannot accept a quadratic objective unchanged. Fork/adapt the recursive
pattern; do not force the quadratic into a lifted or approximated LP.

## Required solver contract

### Inputs

- `q : vec3`: preferred unconstrained point in a normalized local frame.
- `planes[0:m]`, each `plane = {a: vec3, b: scalar, id: integer}` representing
  `dot(a,p) >= b`.
- `options` containing:
  - deterministic shuffle seed;
  - relative feasibility tolerance;
  - rank/parallel tolerance;
  - optional conservative outward offset;
  - maximum supported plane count (only for fixed scratch storage).

### Outputs

- `status`: `success`, `infeasible`, `numerical_failure`, or
  `capacity_exceeded`.
- `p : vec3` on success.
- `active_ids[3]` and `active_count` containing an independent support set.
- `max_violation = max_i(b_i-dot(a_i,p))`.
- Optional KKT multipliers for the returned active planes.
- A diagnostic bitfield: redundant planes seen, rank drop, fallback used,
  conservative offset applied.

### Success conditions

On `success`:

1. `p` is feasible to the requested tolerance.
2. `p` minimizes the regularized objective among feasible points to the test
   tolerance.
3. The active support is independent, has cardinality at most 3, and its KKT
   multipliers are nonnegative to tolerance.

An empty feasible set must never be silently converted into a placement.

## Coordinate and scaling policy

All placement solves must use a local frame:

1. Set origin `c` to the current edge midpoint.
2. Choose scale `s` from a robust local edge-ring length, for example median
   incident edge length, clamped away from zero.
3. Solve for `x=(p-c)/s`.
4. Normalize every nonzero plane normal: `a <- a/||a||`, `b <- b/||a||`.
5. Express `q` and any regularization weight in this normalized frame.

This makes tolerances scale-aware and prevents a small regularization weight
from producing an enormous, poorly represented `q` in world coordinates.

Use `double` throughout the initial implementation. Use `long double` in the
test oracle when available; it is useful as a differential-testing reference,
not as the production containment policy.

## Core algorithm

Represent the current affine subspace as

\[
  p = o + U z,
\]

where columns of `U` are orthonormal and `z` has current dimension `d`.
Initially, `o=0`, `U=I`, and `d=3` in local coordinates.

`project_recursive(q, planes[0:k], o, U, d)` returns the closest feasible
point to `q` within that affine subspace.

### Base case

For no processed planes, return the orthogonal projection of `q` onto the
current affine subspace:

\[
  p=o+UU^T(q-o).
\]

For `d=0`, test all processed plane inequalities at the sole point `o` and
return either `o` or `infeasible`.

### Incremental step

Process planes in a fixed pseudorandom permutation.

1. Recursively/currently solve the first `i-1` planes and obtain `p`.
2. If `dot(a_i,p) >= b_i - feasibility_tolerance`, retain `p`.
3. Otherwise, the optimum after adding plane `i` lies on its boundary. Restrict
   the subspace to
   
   \[
     a_i^T(o+Uz)=b_i.
   \]
4. Let `alpha = U^T a_i` and `beta = b_i-dot(a_i,o)`.
   - If `||alpha||` is below the parallel tolerance, this constraint is either
     redundant (`dot(a_i,o) >= b_i`) or makes the subproblem infeasible.
   - Otherwise choose `z0 = beta*alpha/dot(alpha,alpha)`, set
     `o_new=o+U*z0`, and compute an orthonormal basis `N` for the nullspace of
     `alpha^T`.
   - Recurse on the earlier planes using `o_new`, `U_new=U*N`, and `d-1`.

The key fact is that a newly violated halfspace is active at the new projection;
this justifies the dimension reduction exactly as in fixed-dimensional LP.

### Small-dimensional basis routines

Implement explicit, tested routines rather than a general QR dependency:

- `null_basis_3_to_2(unit_normal)`;
- `null_basis_2_to_1(unit_normal)`;
- normalized cross-product construction for the final one-dimensional basis;
- robust fallback axis selection when a normal is near a coordinate axis.

Always re-orthogonalize and check norms before division. Preserve plane ids along
the recursion so the support witness can be reconstructed.

## Optional production accelerator: warm-start active set

After the recursive solver is correct, optionally add a deterministic primal or
dual active-set solver. Cache a previous active plane set for each surviving
candidate edge and attempt its equality-constrained QP first. On failure,
fallback to the recursive solver.

Do not make the active-set implementation the only initial solver: dependent
planes, nearly parallel planes, and multiplier sign handling are easier to get
wrong than the recursive reference path.

## Independent oracle solver

Implement an exhaustive support-set oracle before optimizing the primary
solver. It enumerates all subsets of the planes of cardinality 0 through 3.

For each independent subset `S`:

1. Solve the equality-constrained projection
   
   \[
     \min_p\|p-q\|^2 \quad\text{s.t.}\quad A_Sp=b_S.
   \]
2. Reject rank-deficient subsets unless a lower-rank equivalent has already
   been considered.
3. Test all inequalities.
4. Retain the feasible candidate with smallest objective.
5. Check active multipliers before accepting it as an optimal KKT candidate.

This has roughly `O(m^4)` scalar checks in 3D. It is intentionally too slow for
the main simplifier but excellent for unit tests, fuzzing, and failure dumps.

## Numerical and containment policy

### Tolerances

After local normalization, begin with separate configurable tolerances:

- `eps_rank`: detects parallel/dependent constraints;
- `eps_feasible`: accepts tiny negative residuals in the numerical solve;
- `eps_kkt`: accepts tiny negative support multipliers;
- `eps_outward`: optional positive RHS offset for conservative production use.

Do not use one global epsilon for all four meanings. Log all residuals in test
failures, then select defaults from observed normalized distributions.

### Conservative placement

For an outer hull, a tiny infeasibility can violate the nesting claim. The
production policy should therefore be one of:

1. solve `a_i^T p >= b_i + eps_outward` directly; or
2. verify, then rerun/reproject with that offset if the unoffset solution has an
   unacceptable residual.

The outward direction must be validated by small known examples before using it.
This is separate from objective regularization.

### Degenerate geometry

The placement stage must return a status, not guess, for:

- zero-area source faces or zero plane normals;
- empty halfspace intersection;
- a local scale below the configured minimum;
- capacity overflow;
- NaN or infinity in inputs or intermediate values.

The mesh simplifier should mark such an edge collapse illegal or route it to an
explicitly configured fallback.

## Integration requirements

1. Reproduce the existing halfspace construction exactly before changing
   placement behavior.
2. Compute the linear volume gradient `g` and regularization reference `p0` in
   the same local frame used by the solver.
3. Use the regularized objective value for edge priority only if that is the
   intended simplification metric; otherwise report both volume increase and
   regularized placement energy explicitly.
4. Keep geometric legality tests outside the QP and run them after placement.
5. Store solver diagnostics in a debug build so a bad collapse can be replayed
   from planes, `q`, seed, scale, and tolerances alone.

## Test plan

### A. Deterministic unit tests

- No planes: return `q`.
- One plane: ordinary point-to-plane projection.
- Two orthogonal planes: projection to their intersection line.
- Three orthogonal planes: projection to their intersection point.
- Preferred point already feasible: return it unchanged.
- Redundant duplicate plane.
- Parallel consistent planes.
- Parallel contradictory planes: `infeasible`.
- Wedge, slab, cone, and bounded box cases.
- Rank-1 and rank-2 active solutions.
- Near-parallel planes and nearly coincident offsets.
- Translation and uniform-scale invariance in world coordinates.
- Known outer-hull local configurations with hand-checked feasible placement.

### B. Differential tests against the exhaustive oracle

Generate thousands of random feasible and infeasible plane sets in normalized
coordinates. For each feasible case compare:

- solver status;
- placement distance/objective;
- maximum constraint violation;
- active-set cardinality and multiplier signs;
- invariance under random plane permutation, using separate fixed shuffle seeds.

On discrepancy, serialize a minimal replay fixture containing `q`, planes,
tolerances, seed, recursive trace, and oracle result.

### C. Metamorphic tests

- Add a redundant feasible plane: result must not change beyond tolerance.
- Duplicate a plane many times: result must not change.
- Apply a rigid transformation and scale, then map result back.
- Increase `eps_outward`: every resulting residual must move outward
  monotonically.
- Increase regularization weight: placement should move continuously toward
  `p0` when no active-set transition occurs.
- Let regularization become large: result approaches projection of `p0`.

### D. Mesh-level regression tests

Build a corpus of closed, consistently oriented meshes: convex shapes, concave
shapes, thin shells, high-valence poles, repeated/near-coplanar tessellation,
scanned meshes, and deliberately bad inputs.

For every accepted collapse:

- validate all local outward-plane residuals;
- run the existing nonintersection and topological legality tests;
- periodically compare global winding/containment samples against the preceding
  mesh;
- record volume change, active count, feasibility margin, and fallback rate.

Run the same corpus through the prior implementation and compare simplification
trajectory, accepted-collapse count, volume growth, and failure modes. Exact
vertex trajectories need not match because the regularized objective creates a
unique but different placement model than a pure LP.

### E. Stress and robustness tests

- randomized coordinate magnitudes from very small to very large before
  normalization;
- adversarial nearly coplanar incident faces;
- plane sets with active sets changing under perturbations near machine epsilon;
- repeated runs with identical seed: bitwise-stable results where the target
  platform supports it;
- sanitizers and NaN/Inf injection;
- performance benchmarks grouped by plane count and active-set dimension.

## Acceptance criteria

The placement solver is ready for default use when:

1. It agrees with the exhaustive oracle on the defined randomized corpus within
   documented normalized tolerances.
2. Every successful result passes independent feasibility and KKT checks.
3. Empty and degenerate cases produce explicit non-success statuses.
4. Mesh-level runs show no unexplained local halfspace violations or nesting
   failures under the conservative placement policy.
5. Median and tail placement times beat the exhaustive oracle by a material
   margin on realistic vertex-ring valences.
6. Failure fixtures are reproducible from serialized solver inputs.

## Suggested development sequence

1. Implement normalized plane construction and a standalone problem fixture
   format.
2. Implement the exhaustive support-set oracle and its KKT/feasibility checker.
3. Implement the recursive 3D/2D/1D projection solver.
4. Add deterministic and randomized differential tests.
5. Integrate behind a debug comparison flag in progressive hulls.
6. Establish conservative offset defaults from corpus data.
7. Benchmark; only then consider warm-start active sets or general SPD `Q`.

## Open decisions to resolve during planning

- Exact regularizer used by the target implementation: isotropic midpoint
  penalty versus a full SPD metric.
- Definition and units of the regularization weight after local scaling.
- Whether priority cost is true volume increase, the regularized objective, or
  a separate error metric.
- Desired conservative margin and whether it is unconditional or retry-only.
- Maximum local valence/scratch capacity and behavior on overflow.
- Whether deterministic reproducibility across CPU architectures is required.
