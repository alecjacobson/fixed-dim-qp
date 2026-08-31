// Exhaustive support-set oracle for the halfspace-projection QP
//
//   min_p ||p-q||^2  s.t.  A p >= b
//
// Deliberately not part of the production igl:: API: it is O(m^Dim) and
// exists purely so the fast recursive solver has a trustworthy, easy-to-audit
// reference to be checked against in unit and differential tests. See
// progressive_hulls_fixed_dim_qp_design.md, "Independent oracle solver".
#ifndef FIXED_DIM_QP_TESTS_HALFSPACE_PROJECTION_ORACLE_H
#define FIXED_DIM_QP_TESTS_HALFSPACE_PROJECTION_ORACLE_H

#include <igl/halfspace_projection_types.h>
#include <Eigen/Dense>
#include <vector>
#include <limits>
#include <cmath>
#include <functional>

namespace fixed_dim_qp_oracle
{
  template <typename Scalar, int Dim>
  struct OracleResult
  {
    igl::HalfspaceProjectionStatus status = igl::HalfspaceProjectionStatus::NUMERICAL_FAILURE;
    Eigen::Matrix<Scalar,Dim,1> p = Eigen::Matrix<Scalar,Dim,1>::Zero();
    std::vector<int> active_ids;
    std::vector<Scalar> multipliers;
    Scalar max_violation = Scalar(0);
    Scalar objective = Scalar(0);
    bool kkt_valid = false;
  };

  namespace internal
  {
    // Calls visit(combo) for every k-combination of {0,...,m-1} in increasing
    // index order.
    template <typename Visit>
    void for_each_combination(int m, int k, Visit visit)
    {
      std::vector<int> combo(k);
      std::function<void(int,int)> rec = [&](int start, int chosen)
      {
        if(chosen == k) { visit(combo); return; }
        for(int i = start;i <= m - (k - chosen);i++)
        {
          combo[chosen] = i;
          rec(i + 1, chosen + 1);
        }
      };
      rec(0, 0);
    }
  }

  /// Brute-force reference solver: enumerate every 0..Dim-cardinality subset
  /// of planes, solve the equality-constrained projection on each
  /// independent subset, keep the feasible candidate with smallest
  /// objective.
  template <typename Scalar, int Dim>
  OracleResult<Scalar,Dim> oracle_project_to_halfspace_intersection(
    const Eigen::Matrix<Scalar,Dim,1> & q,
    const Eigen::Matrix<Scalar,Eigen::Dynamic,Dim> & A,
    const Eigen::Matrix<Scalar,Eigen::Dynamic,1> & b,
    const igl::HalfspaceProjectionOptions<Scalar> & options)
  {
    OracleResult<Scalar,Dim> out;
    const Eigen::Index m = A.rows();

    if(!q.allFinite() || !A.allFinite() || !b.allFinite())
    {
      out.status = igl::HalfspaceProjectionStatus::NUMERICAL_FAILURE;
      return out;
    }
    if(options.max_planes != 0 && static_cast<size_t>(m) > options.max_planes)
    {
      out.status = igl::HalfspaceProjectionStatus::CAPACITY_EXCEEDED;
      return out;
    }

    const Eigen::Matrix<Scalar,Eigen::Dynamic,1> b_offset =
      b.array() + options.eps_outward;

    bool has_best = false;
    Scalar best_obj = std::numeric_limits<Scalar>::infinity();

    const int max_k = static_cast<int>(std::min<Eigen::Index>(Dim, m));
    for(int k = 0;k <= max_k;k++)
    {
      internal::for_each_combination(static_cast<int>(m), k,
        [&](const std::vector<int> & combo)
      {
        Eigen::Matrix<Scalar,Dim,1> p;
        std::vector<Scalar> mu(k, Scalar(0));

        if(k == 0)
        {
          p = q;
        }
        else
        {
          Eigen::Matrix<Scalar,Eigen::Dynamic,Dim> A_S(k, Dim);
          Eigen::Matrix<Scalar,Eigen::Dynamic,1> b_S(k);
          for(int r = 0;r < k;r++)
          {
            A_S.row(r) = A.row(combo[r]);
            b_S(r) = b_offset(combo[r]);
          }
          // Rank check on A_S via smallest singular value.
          Eigen::JacobiSVD<Eigen::Matrix<Scalar,Eigen::Dynamic,Dim>> svd(A_S);
          const Scalar smallest_sv = svd.singularValues()(svd.singularValues().size() - 1);
          if(smallest_sv <= options.eps_rank) return; // rank-deficient: skip

          const Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> gram = A_S * A_S.transpose();
          const Eigen::Matrix<Scalar,Eigen::Dynamic,1> rhs = b_S - A_S * q;
          const Eigen::Matrix<Scalar,Eigen::Dynamic,1> mu_vec = gram.ldlt().solve(rhs);
          for(int r = 0;r < k;r++) mu[r] = mu_vec(r);
          p = q + A_S.transpose() * mu_vec;
        }

        if(!p.allFinite()) return;

        // Feasibility w.r.t. the *full*, offset constraint set.
        const Eigen::Matrix<Scalar,Eigen::Dynamic,1> residual = b_offset - A * p; // want <= eps_feasible
        const Scalar max_violation = m == 0 ? Scalar(0) : residual.maxCoeff();
        if(max_violation > options.eps_feasible) return;

        const Scalar objective = (p - q).squaredNorm();
        if(!has_best || objective < best_obj)
        {
          has_best = true;
          best_obj = objective;
          out.p = p;
          out.active_ids = combo;
          out.multipliers = mu;
          out.max_violation = max_violation;
          out.objective = objective;
          out.kkt_valid = true;
          for(int r = 0;r < k;r++)
          {
            if(mu[r] < -options.eps_kkt) { out.kkt_valid = false; break; }
          }
        }
      });
    }

    out.status = has_best
      ? igl::HalfspaceProjectionStatus::SUCCESS
      : igl::HalfspaceProjectionStatus::INFEASIBLE;
    return out;
  }
}

#endif
