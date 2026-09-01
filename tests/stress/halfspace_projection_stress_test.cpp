// Section E stress/robustness tests. See
// progressive_hulls_fixed_dim_qp_design.md, "Stress and robustness tests".
#include <catch2/catch_test_macros.hpp>
#include <igl/project_to_halfspace_intersection.h>
#include <random>
#include <cmath>
#include <limits>

using igl::HalfspaceProjectionOptions;
using igl::HalfspaceProjectionResult;
using igl::HalfspaceProjectionStatus;

namespace
{
  HalfspaceProjectionOptions<double> options_for_scale(double scale, uint64_t seed)
  {
    HalfspaceProjectionOptions<double> o;
    // Tolerances scale with the problem magnitude, matching the design
    // spec's normalization guidance -- these tests intentionally probe
    // *unnormalized* magnitude ranges, so the tolerances must follow.
    o.eps_rank = std::max(1e-12, 1e-9 * scale);
    o.eps_feasible = std::max(1e-12, 1e-8 * scale);
    o.eps_kkt = std::max(1e-10, 1e-6 * scale);
    o.eps_outward = 0;
    o.seed = seed;
    return o;
  }
}

TEST_CASE("stress: randomized coordinate magnitudes 1e-6 to 1e6 never crash or return garbage", "[stress]")
{
  std::mt19937_64 rng(0xF00DBEEF);
  std::uniform_int_distribution<int> m_dist(0, 12);
  std::normal_distribution<double> normal_dist(0.0, 1.0);

  for(int scale_exp = -6;scale_exp <= 6;scale_exp += 2)
  {
    const double scale = std::pow(10.0, scale_exp);
    std::uniform_real_distribution<double> coord(-scale, scale);

    for(int trial = 0;trial < 100;trial++)
    {
      const int m = m_dist(rng);
      Eigen::Vector3d q(coord(rng), coord(rng), coord(rng));
      Eigen::Matrix<double,Eigen::Dynamic,3> A(m,3);
      Eigen::VectorXd b(m);
      for(int i = 0;i < m;i++)
      {
        Eigen::Vector3d n(normal_dist(rng), normal_dist(rng), normal_dist(rng));
        if(n.norm() < 1e-9) n = Eigen::Vector3d(1,0,0);
        n.normalize();
        A.row(i) = n;
        b(i) = coord(rng);
      }

      HalfspaceProjectionResult<double,3> r;
      const auto status = igl::project_to_halfspace_intersection<double,3>(
        q, A, b, options_for_scale(scale, static_cast<uint64_t>(trial)), r);

      INFO("scale=" << scale << " trial=" << trial << " m=" << m);
      REQUIRE(status != HalfspaceProjectionStatus::CAPACITY_EXCEEDED);
      if(status == HalfspaceProjectionStatus::SUCCESS)
      {
        REQUIRE(r.p.allFinite());
        REQUIRE(std::isfinite(r.max_violation));
        REQUIRE(r.active_count >= 0);
        REQUIRE(r.active_count <= 3);
      }
    }
  }
}

TEST_CASE("stress: adversarial nearly-coplanar planes never crash or return garbage", "[stress]")
{
  std::mt19937_64 rng(0xC0A1E5CE);
  std::normal_distribution<double> jitter(0.0, 1.0);

  for(int trial = 0;trial < 500;trial++)
  {
    // A base plane, plus several near-duplicates perturbed at ~machine
    // epsilon to double-precision-relative scale.
    Eigen::Vector3d n0(jitter(rng), jitter(rng), jitter(rng));
    if(n0.norm() < 1e-9) n0 = Eigen::Vector3d(0,0,1);
    n0.normalize();
    const double b0 = jitter(rng);

    std::uniform_real_distribution<double> eps_dist(1e-16, 1e-8);
    const int extra = 5;
    Eigen::Matrix<double,Eigen::Dynamic,3> A(1 + extra, 3);
    Eigen::VectorXd b(1 + extra);
    A.row(0) = n0;
    b(0) = b0;
    for(int k = 0;k < extra;k++)
    {
      const double eps = eps_dist(rng);
      Eigen::Vector3d n = n0 + eps * Eigen::Vector3d(jitter(rng), jitter(rng), jitter(rng));
      n.normalize();
      A.row(1 + k) = n;
      b(1 + k) = b0 + eps * jitter(rng);
    }

    Eigen::Vector3d q(jitter(rng), jitter(rng), jitter(rng));
    HalfspaceProjectionOptions<double> o;
    o.eps_rank = 1e-9;
    o.eps_feasible = 1e-8;
    o.eps_kkt = 1e-6;
    o.seed = static_cast<uint64_t>(trial);

    HalfspaceProjectionResult<double,3> r;
    const auto status = igl::project_to_halfspace_intersection<double,3>(q, A, b, o, r);

    INFO("trial=" << trial);
    REQUIRE(status != HalfspaceProjectionStatus::CAPACITY_EXCEEDED);
    if(status == HalfspaceProjectionStatus::SUCCESS)
    {
      REQUIRE(r.p.allFinite());
    }
  }
}

TEST_CASE("stress: NaN/Inf injected into inputs always yields NUMERICAL_FAILURE, never a crash", "[stress]")
{
  std::mt19937_64 rng(0xDEAD10CC);
  std::normal_distribution<double> jitter(0.0, 1.0);
  const double nan_v = std::numeric_limits<double>::quiet_NaN();
  const double inf_v = std::numeric_limits<double>::infinity();

  for(int trial = 0;trial < 200;trial++)
  {
    const int m = 3 + static_cast<int>(trial % 5);
    Eigen::Vector3d q(jitter(rng), jitter(rng), jitter(rng));
    Eigen::Matrix<double,Eigen::Dynamic,3> A(m,3);
    Eigen::VectorXd b(m);
    for(int i = 0;i < m;i++)
    {
      Eigen::Vector3d n(jitter(rng), jitter(rng), jitter(rng));
      if(n.norm() < 1e-9) n = Eigen::Vector3d(1,0,0);
      n.normalize();
      A.row(i) = n;
      b(i) = jitter(rng);
    }

    // Inject exactly one bad value into q, A, or b.
    const int target = trial % 3;
    const double bad = (trial % 2 == 0) ? nan_v : inf_v;
    if(target == 0) q(trial % 3) = bad;
    else if(target == 1) A(trial % m, trial % 3) = bad;
    else b(trial % m) = bad;

    HalfspaceProjectionOptions<double> o;
    o.seed = static_cast<uint64_t>(trial);
    HalfspaceProjectionResult<double,3> r;
    const auto status = igl::project_to_halfspace_intersection<double,3>(q, A, b, o, r);

    INFO("trial=" << trial << " target=" << target);
    REQUIRE(status == HalfspaceProjectionStatus::NUMERICAL_FAILURE);
  }
}

TEST_CASE("stress: identical seed gives bitwise-identical results across repeated runs", "[stress]")
{
  std::mt19937_64 rng(0xB16B00B5);
  std::normal_distribution<double> jitter(0.0, 1.0);

  for(int trial = 0;trial < 50;trial++)
  {
    const int m = 1 + static_cast<int>(trial % 15);
    Eigen::Vector3d q(jitter(rng), jitter(rng), jitter(rng));
    Eigen::Matrix<double,Eigen::Dynamic,3> A(m,3);
    Eigen::VectorXd b(m);
    for(int i = 0;i < m;i++)
    {
      Eigen::Vector3d n(jitter(rng), jitter(rng), jitter(rng));
      if(n.norm() < 1e-9) n = Eigen::Vector3d(1,0,0);
      n.normalize();
      A.row(i) = n;
      b(i) = jitter(rng);
    }

    HalfspaceProjectionOptions<double> o;
    o.seed = 0xA11CE;

    HalfspaceProjectionResult<double,3> r1, r2, r3;
    const auto s1 = igl::project_to_halfspace_intersection<double,3>(q, A, b, o, r1);
    const auto s2 = igl::project_to_halfspace_intersection<double,3>(q, A, b, o, r2);
    const auto s3 = igl::project_to_halfspace_intersection<double,3>(q, A, b, o, r3);

    REQUIRE(s1 == s2);
    REQUIRE(s2 == s3);
    // Bitwise, not approximate: same seed/inputs must retrace the exact
    // same recursion path and floating-point operation order.
    REQUIRE(r1.p == r2.p);
    REQUIRE(r2.p == r3.p);
    REQUIRE(r1.active_count == r2.active_count);
    REQUIRE(r1.max_violation == r2.max_violation);
  }
}
