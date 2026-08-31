// Section B differential tests: randomized plane sets checked against the
// exhaustive oracle. See progressive_hulls_fixed_dim_qp_design.md.
#include <catch2/catch_test_macros.hpp>
#include <igl/project_to_halfspace_intersection.h>
#include <igl/halfspace_projection_fixture.h>
#include "../oracle/halfspace_projection_oracle.h"
#include <random>
#include <sstream>
#include <algorithm>
#include <numeric>

using igl::HalfspaceProjectionOptions;
using igl::HalfspaceProjectionResult;
using igl::HalfspaceProjectionStatus;
using igl::HalfspaceProblem;

namespace
{
  struct RandomCase
  {
    Eigen::Vector3d q;
    Eigen::Matrix<double,Eigen::Dynamic,3> A;
    Eigen::VectorXd b;
  };

  RandomCase make_random_case(std::mt19937_64 & rng, int m, double scale)
  {
    std::uniform_real_distribution<double> coord(-scale, scale);
    std::normal_distribution<double> normal_dist(0.0, 1.0);

    RandomCase c;
    c.q = Eigen::Vector3d(coord(rng), coord(rng), coord(rng));
    c.A.resize(m, 3);
    c.b.resize(m);
    for(int i = 0;i < m;i++)
    {
      Eigen::Vector3d n(normal_dist(rng), normal_dist(rng), normal_dist(rng));
      if(n.norm() < 1e-6) n = Eigen::Vector3d(1,0,0);
      n.normalize();
      c.A.row(i) = n;
      c.b(i) = coord(rng);
    }
    return c;
  }

  HalfspaceProjectionOptions<double> differential_options(uint64_t seed)
  {
    HalfspaceProjectionOptions<double> o;
    o.eps_rank = 1e-9;
    o.eps_feasible = 1e-8;
    o.eps_kkt = 1e-7;
    o.eps_outward = 0;
    o.seed = seed;
    return o;
  }

  std::string dump_fixture(
    const Eigen::Vector3d & q,
    const Eigen::Matrix<double,Eigen::Dynamic,3> & A,
    const Eigen::VectorXd & b,
    const HalfspaceProjectionOptions<double> & options)
  {
    HalfspaceProblem<double,3> problem;
    problem.q = q;
    problem.A = A;
    problem.b = b;
    problem.options = options;
    std::ostringstream ss;
    igl::write_halfspace_projection_fixture(ss, problem);
    return ss.str();
  }
}

TEST_CASE("differential: solver matches oracle on random plane sets", "[differential][section-b]")
{
  std::mt19937_64 rng(0xD1FF7E57);
  std::uniform_int_distribution<int> m_dist(0, 12);
  const int trials = 3000;
  const double scale = 5.0;

  int n_success = 0, n_infeasible = 0;

  for(int t = 0;t < trials;t++)
  {
    const int m = m_dist(rng);
    const RandomCase c = make_random_case(rng, m, scale);
    const auto options = differential_options(0xA5A5 + static_cast<uint64_t>(t));

    HalfspaceProjectionResult<double,3> solver_result;
    const auto solver_status = igl::project_to_halfspace_intersection<double,3>(
      c.q, c.A, c.b, options, solver_result);

    const auto oracle_result = fixed_dim_qp_oracle::oracle_project_to_halfspace_intersection<double,3>(
      c.q, c.A, c.b, options);

    INFO("trial " << t << " m=" << m);
    INFO(dump_fixture(c.q, c.A, c.b, options));

    REQUIRE(solver_status == oracle_result.status);

    if(oracle_result.status == HalfspaceProjectionStatus::SUCCESS)
    {
      n_success++;
      REQUIRE(solver_result.p.isApprox(oracle_result.p, 1e-6));
      const double solver_obj = (solver_result.p - c.q).squaredNorm();
      REQUIRE(std::abs(solver_obj - oracle_result.objective) < 1e-6 * std::max(1.0, oracle_result.objective));
      REQUIRE(solver_result.active_count <= 3);
      REQUIRE(solver_result.max_violation <= 1e-6);
    }
    else
    {
      n_infeasible++;
    }
  }

  // Sanity: the random generator should exercise both branches meaningfully.
  REQUIRE(n_success > 0);
  REQUIRE(n_infeasible > 0);
}

TEST_CASE("differential: solver result invariant under plane permutation", "[differential][section-b]")
{
  std::mt19937_64 rng(0x9E3779B9);
  std::uniform_int_distribution<int> m_dist(1, 10);
  const int trials = 500;
  const double scale = 5.0;

  for(int t = 0;t < trials;t++)
  {
    const int m = m_dist(rng);
    const RandomCase c = make_random_case(rng, m, scale);
    const auto options = differential_options(0xB00B + static_cast<uint64_t>(t));

    HalfspaceProjectionResult<double,3> r1;
    const auto s1 = igl::project_to_halfspace_intersection<double,3>(c.q, c.A, c.b, options, r1);

    std::vector<int> perm(m);
    std::iota(perm.begin(), perm.end(), 0);
    std::mt19937_64 perm_rng(0xC0FFEE + static_cast<uint64_t>(t));
    std::shuffle(perm.begin(), perm.end(), perm_rng);

    Eigen::Matrix<double,Eigen::Dynamic,3> A2(m,3);
    Eigen::VectorXd b2(m);
    for(int i = 0;i < m;i++)
    {
      A2.row(i) = c.A.row(perm[i]);
      b2(i) = c.b(perm[i]);
    }

    HalfspaceProjectionResult<double,3> r2;
    const auto s2 = igl::project_to_halfspace_intersection<double,3>(c.q, A2, b2, options, r2);

    INFO("trial " << t << " m=" << m);
    INFO(dump_fixture(c.q, c.A, c.b, options));
    REQUIRE(s1 == s2);
    if(s1 == HalfspaceProjectionStatus::SUCCESS)
    {
      REQUIRE(r1.p.isApprox(r2.p, 1e-6));
    }
  }
}

TEST_CASE("differential: solver result invariant under seed choice", "[differential][section-b]")
{
  std::mt19937_64 rng(0x1234ABCD);
  std::uniform_int_distribution<int> m_dist(1, 10);
  const int trials = 500;
  const double scale = 5.0;

  for(int t = 0;t < trials;t++)
  {
    const int m = m_dist(rng);
    const RandomCase c = make_random_case(rng, m, scale);

    auto options_a = differential_options(1);
    auto options_b = differential_options(999999);

    HalfspaceProjectionResult<double,3> ra, rb;
    const auto sa = igl::project_to_halfspace_intersection<double,3>(c.q, c.A, c.b, options_a, ra);
    const auto sb = igl::project_to_halfspace_intersection<double,3>(c.q, c.A, c.b, options_b, rb);

    INFO("trial " << t << " m=" << m);
    INFO(dump_fixture(c.q, c.A, c.b, options_a));
    REQUIRE(sa == sb);
    if(sa == HalfspaceProjectionStatus::SUCCESS)
    {
      REQUIRE(ra.p.isApprox(rb.p, 1e-6));
    }
  }
}
