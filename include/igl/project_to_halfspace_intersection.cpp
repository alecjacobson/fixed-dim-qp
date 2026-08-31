#include "project_to_halfspace_intersection.h"
#include <Eigen/Dense>
#include <vector>
#include <array>
#include <numeric>
#include <algorithm>
#include <random>
#include <cmath>

namespace igl
{
  namespace internal
  {
    // Orthonormal basis (d x (d-1)) of the orthogonal complement of a unit
    // vector in R^d, for any d >= 1 (d==1 yields a d x 0 empty basis). Built
    // on Eigen's Householder QR rather than hand-specialized 3D/2D formulas:
    // at these sizes (d <= Dim <= a handful) the cost is negligible and the
    // reflector construction is already robust to the normal being close to
    // any particular coordinate axis, which is exactly the "robust fallback
    // axis selection" the design spec asks for.
    template <typename Scalar>
    IGL_INLINE Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> orthonormal_complement(
      const Eigen::Matrix<Scalar,Eigen::Dynamic,1> & unit_vec)
    {
      const Eigen::Index d = unit_vec.size();
      Eigen::HouseholderQR<Eigen::Matrix<Scalar,Eigen::Dynamic,1>> qr(unit_vec);
      const Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> Q = qr.householderQ();
      return Q.rightCols(d - 1);
    }

    // Recursive Seidel/SDLP-style projection of q onto the intersection of
    // the first `count` halfspaces named by `ids` (a fixed permutation
    // shared by every recursion level -- only the prefix *length* shrinks on
    // dimension reduction, so it's passed as a range rather than copied),
    // restricted to the affine subspace {o + U*z : z in R^d}. See
    // project_to_halfspace_intersection.h and
    // progressive_hulls_fixed_dim_qp_design.md, "Core algorithm".
    //
    // active_out/active_count use a fixed Dim-sized scratch array rather
    // than a heap-allocated container: the active support never exceeds Dim
    // by construction (d strictly decreases by one per activation and
    // starts at Dim), so this and the ids-by-range change above keep the
    // whole recursion allocation-free except for the small per-level
    // Eigen::Dynamic temporaries (alpha/N/U_new), whose size is bounded by
    // the current dimension d <= Dim.
    template <typename Scalar, int Dim>
    IGL_INLINE bool halfspace_projection_recurse(
      const Eigen::Matrix<Scalar,Dim,1> & q,
      const Eigen::Matrix<Scalar,Eigen::Dynamic,Dim> & A,
      const Eigen::Matrix<Scalar,Eigen::Dynamic,1> & b,
      const HalfspaceProjectionOptions<Scalar> & options,
      const std::vector<int> & ids,
      const int count,
      const Eigen::Matrix<Scalar,Dim,1> & o,
      const Eigen::Matrix<Scalar,Dim,Eigen::Dynamic> & U,
      int d,
      uint32_t & diagnostics,
      Eigen::Matrix<Scalar,Dim,1> & p_out,
      std::array<int,Dim> & active_out,
      int & active_count)
    {
      if(d == 0)
      {
        for(int idx = 0;idx < count;idx++)
        {
          const int id = ids[idx];
          const Scalar lhs = A.row(id) * o;
          if(lhs < b(id) - options.eps_feasible) return false;
        }
        p_out = o;
        active_count = 0;
        return true;
      }

      p_out = o + U * (U.transpose() * (q - o));
      active_count = 0;

      for(int idx = 0;idx < count;idx++)
      {
        const int i = ids[idx];
        const Scalar lhs = A.row(i) * p_out;
        if(lhs >= b(i) - options.eps_feasible) continue; // not active

        const Eigen::Matrix<Scalar,Eigen::Dynamic,1> alpha = U.transpose() * A.row(i).transpose();
        const Scalar beta = b(i) - static_cast<Scalar>(A.row(i) * o);
        const Scalar alpha_sq = alpha.squaredNorm();
        const Scalar alpha_norm = std::sqrt(alpha_sq);

        if(alpha_norm <= options.eps_rank)
        {
          // a_i is (numerically) orthogonal to every remaining free
          // direction, so a_i^T p is constant (== a_i^T o) throughout this
          // subspace: either already-satisfied-elsewhere-but-flagged-here
          // (redundant, tolerate) or genuinely unsatisfiable here (infeasible).
          const Scalar lhs_o = A.row(i) * o;
          if(lhs_o >= b(i) - options.eps_feasible)
          {
            diagnostics |= IGL_HSP_DIAG_REDUNDANT_PLANE_SEEN;
            continue;
          }
          diagnostics |= IGL_HSP_DIAG_RANK_DROP;
          return false;
        }

        const Eigen::Matrix<Scalar,Eigen::Dynamic,1> z0 = (beta / alpha_sq) * alpha;
        const Eigen::Matrix<Scalar,Dim,1> o_new = o + U * z0;
        const Eigen::Matrix<Scalar,Eigen::Dynamic,1> alpha_unit = alpha / alpha_norm;
        const Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> N = orthonormal_complement<Scalar>(alpha_unit);
        const Eigen::Matrix<Scalar,Dim,Eigen::Dynamic> U_new = U * N;

        Eigen::Matrix<Scalar,Dim,1> p2;
        std::array<int,Dim> active2;
        int active2_count = 0;
        if(!halfspace_projection_recurse<Scalar,Dim>(
             q, A, b, options, ids, idx, o_new, U_new, d - 1, diagnostics, p2, active2, active2_count))
        {
          return false;
        }
        p_out = p2;
        active_out = active2;
        active_count = active2_count;
        active_out[active_count++] = i;
      }
      return true;
    }
  }
}

template <typename Scalar, int Dim>
IGL_INLINE igl::HalfspaceProjectionStatus igl::project_to_halfspace_intersection(
  const Eigen::Matrix<Scalar,Dim,1> & q,
  const Eigen::Matrix<Scalar,Eigen::Dynamic,Dim> & A,
  const Eigen::Matrix<Scalar,Eigen::Dynamic,1> & b,
  const igl::HalfspaceProjectionOptions<Scalar> & options,
  igl::HalfspaceProjectionResult<Scalar,Dim> & result)
{
  result = igl::HalfspaceProjectionResult<Scalar,Dim>();
  const Eigen::Index m = A.rows();

  if(!q.allFinite() || (m > 0 && (!A.allFinite() || !b.allFinite())))
  {
    result.status = igl::HalfspaceProjectionStatus::NUMERICAL_FAILURE;
    return result.status;
  }
  if(options.max_planes != 0 && static_cast<size_t>(m) > options.max_planes)
  {
    result.status = igl::HalfspaceProjectionStatus::CAPACITY_EXCEEDED;
    return result.status;
  }

  const Eigen::Matrix<Scalar,Eigen::Dynamic,1> b_eff = b.array() + options.eps_outward;

  std::vector<int> ids(static_cast<size_t>(m));
  std::iota(ids.begin(), ids.end(), 0);
  std::mt19937_64 rng(options.seed);
  std::shuffle(ids.begin(), ids.end(), rng);

  Eigen::Matrix<Scalar,Dim,1> o = Eigen::Matrix<Scalar,Dim,1>::Zero();
  Eigen::Matrix<Scalar,Dim,Eigen::Dynamic> U(Dim,Dim);
  U.setIdentity();

  uint32_t diagnostics = 0;
  Eigen::Matrix<Scalar,Dim,1> p;
  std::array<int,Dim> active;
  int active_count_raw = 0;
  const bool feasible = igl::internal::halfspace_projection_recurse<Scalar,Dim>(
    q, A, b_eff, options, ids, static_cast<int>(m), o, U, Dim, diagnostics, p, active, active_count_raw);

  if(options.eps_outward > 0) diagnostics |= IGL_HSP_DIAG_CONSERVATIVE_OFFSET_APPLIED;
  result.diagnostics = diagnostics;

  if(!feasible)
  {
    result.status = igl::HalfspaceProjectionStatus::INFEASIBLE;
    return result.status;
  }
  if(!p.allFinite())
  {
    result.status = igl::HalfspaceProjectionStatus::NUMERICAL_FAILURE;
    return result.status;
  }

  result.p = p;
  result.active_count = std::min(active_count_raw, Dim);
  for(int r = 0;r < result.active_count;r++) result.active_ids[r] = active[static_cast<size_t>(r)];

  Scalar max_violation = Scalar(0);
  for(Eigen::Index i = 0;i < m;i++)
  {
    const Scalar v = b(i) - static_cast<Scalar>(A.row(i) * p);
    if(v > max_violation) max_violation = v;
  }
  result.max_violation = max_violation;

  if(result.active_count > 0)
  {
    const int k = result.active_count;
    Eigen::Matrix<Scalar,Eigen::Dynamic,Dim> A_S(k,Dim);
    Eigen::Matrix<Scalar,Eigen::Dynamic,1> b_S(k);
    for(int r = 0;r < k;r++)
    {
      A_S.row(r) = A.row(result.active_ids[r]);
      b_S(r) = b_eff(result.active_ids[r]);
    }
    const Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> gram = A_S * A_S.transpose();
    const Eigen::Matrix<Scalar,Eigen::Dynamic,1> mu = gram.ldlt().solve(b_S - A_S * q);
    for(int r = 0;r < k;r++) result.multipliers[r] = mu(r);

    for(int r = 0;r < k;r++)
    {
      if(result.multipliers[r] < -options.eps_kkt)
      {
        result.status = igl::HalfspaceProjectionStatus::NUMERICAL_FAILURE;
        return result.status;
      }
    }
  }

  result.status = igl::HalfspaceProjectionStatus::SUCCESS;
  return result.status;
}

#ifdef IGL_STATIC_LIBRARY
template igl::HalfspaceProjectionStatus igl::project_to_halfspace_intersection<double,3>(
  const Eigen::Matrix<double,3,1> &,
  const Eigen::Matrix<double,Eigen::Dynamic,3> &,
  const Eigen::Matrix<double,Eigen::Dynamic,1> &,
  const igl::HalfspaceProjectionOptions<double> &,
  igl::HalfspaceProjectionResult<double,3> &);
#endif
