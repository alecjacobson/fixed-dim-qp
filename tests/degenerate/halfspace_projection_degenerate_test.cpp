// Degenerate-input and numerical-policy tests. See
// progressive_hulls_fixed_dim_qp_design.md, "Numerical and containment
// policy" and "Degenerate geometry". Local-scale-below-minimum is a
// caller/integration-layer concern (the local normalization frame is built
// by the caller before invoking this primitive) and is exercised in the
// fixed-dim-qp-bench repo's mesh integration instead.
#include <catch2/catch_test_macros.hpp>
#include <igl/project_to_halfspace_intersection.h>
#include <limits>

using igl::HalfspaceProjectionOptions;
using igl::HalfspaceProjectionResult;
using igl::HalfspaceProjectionStatus;

namespace
{
  HalfspaceProjectionOptions<double> default_options()
  {
    HalfspaceProjectionOptions<double> o;
    o.eps_rank = 1e-9;
    o.eps_feasible = 1e-9;
    o.eps_kkt = 1e-9;
    o.eps_outward = 0;
    o.seed = 3;
    return o;
  }
}

TEST_CASE("degenerate: zero-normal plane that is trivially satisfied is redundant", "[degenerate]")
{
  // 0^T p >= -1 holds everywhere.
  Eigen::Vector3d q(2,3,4);
  Eigen::Matrix<double,Eigen::Dynamic,3> A(1,3);
  A << 0,0,0;
  Eigen::VectorXd b(1); b << -1;
  HalfspaceProjectionResult<double,3> r;
  const auto status = igl::project_to_halfspace_intersection<double,3>(q, A, b, default_options(), r);
  REQUIRE(status == HalfspaceProjectionStatus::SUCCESS);
  REQUIRE(r.p.isApprox(q));
}

TEST_CASE("degenerate: zero-normal plane that is never satisfiable is infeasible", "[degenerate]")
{
  // 0^T p >= 1 never holds.
  Eigen::Vector3d q(2,3,4);
  Eigen::Matrix<double,Eigen::Dynamic,3> A(1,3);
  A << 0,0,0;
  Eigen::VectorXd b(1); b << 1;
  HalfspaceProjectionResult<double,3> r;
  const auto status = igl::project_to_halfspace_intersection<double,3>(q, A, b, default_options(), r);
  REQUIRE(status == HalfspaceProjectionStatus::INFEASIBLE);
}

TEST_CASE("degenerate: zero-normal plane mixed with real constraints", "[degenerate]")
{
  Eigen::Vector3d q(0,0,-5);
  Eigen::Matrix<double,Eigen::Dynamic,3> A(2,3);
  A << 0,0,0,
       0,0,1;
  Eigen::VectorXd b(2); b << -1, 0;
  HalfspaceProjectionResult<double,3> r;
  const auto status = igl::project_to_halfspace_intersection<double,3>(q, A, b, default_options(), r);
  REQUIRE(status == HalfspaceProjectionStatus::SUCCESS);
  REQUIRE(r.p.isApprox(Eigen::Vector3d(0,0,0)));
}

TEST_CASE("degenerate: NaN in q yields NUMERICAL_FAILURE", "[degenerate]")
{
  Eigen::Vector3d q(std::numeric_limits<double>::quiet_NaN(), 0, 0);
  Eigen::Matrix<double,Eigen::Dynamic,3> A(1,3); A << 0,0,1;
  Eigen::VectorXd b(1); b << 0;
  HalfspaceProjectionResult<double,3> r;
  const auto status = igl::project_to_halfspace_intersection<double,3>(q, A, b, default_options(), r);
  REQUIRE(status == HalfspaceProjectionStatus::NUMERICAL_FAILURE);
}

TEST_CASE("degenerate: Inf in a plane normal yields NUMERICAL_FAILURE", "[degenerate]")
{
  Eigen::Vector3d q(0,0,0);
  Eigen::Matrix<double,Eigen::Dynamic,3> A(1,3);
  A << std::numeric_limits<double>::infinity(), 0, 0;
  Eigen::VectorXd b(1); b << 0;
  HalfspaceProjectionResult<double,3> r;
  const auto status = igl::project_to_halfspace_intersection<double,3>(q, A, b, default_options(), r);
  REQUIRE(status == HalfspaceProjectionStatus::NUMERICAL_FAILURE);
}

TEST_CASE("degenerate: NaN in b yields NUMERICAL_FAILURE", "[degenerate]")
{
  Eigen::Vector3d q(0,0,0);
  Eigen::Matrix<double,Eigen::Dynamic,3> A(1,3); A << 0,0,1;
  Eigen::VectorXd b(1); b << std::numeric_limits<double>::quiet_NaN();
  HalfspaceProjectionResult<double,3> r;
  const auto status = igl::project_to_halfspace_intersection<double,3>(q, A, b, default_options(), r);
  REQUIRE(status == HalfspaceProjectionStatus::NUMERICAL_FAILURE);
}

TEST_CASE("degenerate: exceeding max_planes yields CAPACITY_EXCEEDED", "[degenerate]")
{
  Eigen::Vector3d q(0,0,0);
  Eigen::Matrix<double,Eigen::Dynamic,3> A(3,3);
  A << 1,0,0, 0,1,0, 0,0,1;
  Eigen::VectorXd b(3); b << 0,0,0;
  auto o = default_options();
  o.max_planes = 2;
  HalfspaceProjectionResult<double,3> r;
  const auto status = igl::project_to_halfspace_intersection<double,3>(q, A, b, o, r);
  REQUIRE(status == HalfspaceProjectionStatus::CAPACITY_EXCEEDED);
}

TEST_CASE("degenerate: max_planes at exactly the limit still succeeds", "[degenerate]")
{
  Eigen::Vector3d q(0,0,0);
  Eigen::Matrix<double,Eigen::Dynamic,3> A(3,3);
  A << 1,0,0, 0,1,0, 0,0,1;
  Eigen::VectorXd b(3); b << 0,0,0;
  auto o = default_options();
  o.max_planes = 3;
  HalfspaceProjectionResult<double,3> r;
  const auto status = igl::project_to_halfspace_intersection<double,3>(q, A, b, o, r);
  REQUIRE(status == HalfspaceProjectionStatus::SUCCESS);
}

TEST_CASE("degenerate: max_planes 0 means unbounded", "[degenerate]")
{
  Eigen::Vector3d q(0,0,0);
  Eigen::Matrix<double,Eigen::Dynamic,3> A(3,3);
  A << 1,0,0, 0,1,0, 0,0,1;
  Eigen::VectorXd b(3); b << 0,0,0;
  auto o = default_options();
  o.max_planes = 0;
  HalfspaceProjectionResult<double,3> r;
  const auto status = igl::project_to_halfspace_intersection<double,3>(q, A, b, o, r);
  REQUIRE(status == HalfspaceProjectionStatus::SUCCESS);
}

TEST_CASE("degenerate: eps_outward with a genuinely empty polytope stays infeasible", "[degenerate]")
{
  Eigen::Vector3d q(0,0,0);
  Eigen::Matrix<double,Eigen::Dynamic,3> A(2,3);
  A << 0,0,1,
       0,0,-1;
  Eigen::VectorXd b(2); b << 1,1; // z>=1 and z<=-1: empty
  auto o = default_options();
  o.eps_outward = 0.5;
  HalfspaceProjectionResult<double,3> r;
  const auto status = igl::project_to_halfspace_intersection<double,3>(q, A, b, o, r);
  REQUIRE(status == HalfspaceProjectionStatus::INFEASIBLE);
}
