#include <catch2/catch_test_macros.hpp>
#include <igl/project_to_halfspace_intersection.h>

using igl::HalfspaceProjectionOptions;
using igl::HalfspaceProjectionResult;
using igl::HalfspaceProjectionStatus;
using igl::project_to_halfspace_intersection;

namespace
{
  HalfspaceProjectionOptions<double> default_options()
  {
    HalfspaceProjectionOptions<double> o;
    o.eps_rank = 1e-9;
    o.eps_feasible = 1e-9;
    o.eps_kkt = 1e-9;
    o.eps_outward = 0;
    o.seed = 42;
    return o;
  }
}

TEST_CASE("project_to_halfspace_intersection: no planes returns q", "[solver][section-a]")
{
  Eigen::Vector3d q(1,2,3);
  Eigen::Matrix<double,Eigen::Dynamic,3> A(0,3);
  Eigen::VectorXd b(0);
  HalfspaceProjectionResult<double,3> r;
  auto status = project_to_halfspace_intersection<double,3>(q, A, b, default_options(), r);
  REQUIRE(status == HalfspaceProjectionStatus::SUCCESS);
  REQUIRE(r.p.isApprox(q));
  REQUIRE(r.active_count == 0);
}

TEST_CASE("project_to_halfspace_intersection: one plane ordinary projection", "[solver][section-a]")
{
  Eigen::Vector3d q(0,0,-5);
  Eigen::Matrix<double,Eigen::Dynamic,3> A(1,3);
  A << 0,0,1;
  Eigen::VectorXd b(1); b << 0;
  HalfspaceProjectionResult<double,3> r;
  auto status = project_to_halfspace_intersection<double,3>(q, A, b, default_options(), r);
  REQUIRE(status == HalfspaceProjectionStatus::SUCCESS);
  REQUIRE(r.p.isApprox(Eigen::Vector3d(0,0,0)));
  REQUIRE(r.active_count == 1);
}

TEST_CASE("project_to_halfspace_intersection: two orthogonal planes -> line", "[solver][section-a]")
{
  Eigen::Vector3d q(-3,-4,7);
  Eigen::Matrix<double,Eigen::Dynamic,3> A(2,3);
  A << 1,0,0,
       0,1,0;
  Eigen::VectorXd b(2); b << 0,0;
  HalfspaceProjectionResult<double,3> r;
  auto status = project_to_halfspace_intersection<double,3>(q, A, b, default_options(), r);
  REQUIRE(status == HalfspaceProjectionStatus::SUCCESS);
  REQUIRE(r.p.isApprox(Eigen::Vector3d(0,0,7)));
  REQUIRE(r.active_count == 2);
}

TEST_CASE("project_to_halfspace_intersection: three orthogonal planes -> point", "[solver][section-a]")
{
  Eigen::Vector3d q(-1,-2,-3);
  Eigen::Matrix<double,Eigen::Dynamic,3> A(3,3);
  A << 1,0,0,
       0,1,0,
       0,0,1;
  Eigen::VectorXd b(3); b << 0,0,0;
  HalfspaceProjectionResult<double,3> r;
  auto status = project_to_halfspace_intersection<double,3>(q, A, b, default_options(), r);
  REQUIRE(status == HalfspaceProjectionStatus::SUCCESS);
  REQUIRE(r.p.isApprox(Eigen::Vector3d(0,0,0)));
  REQUIRE(r.active_count == 3);
}

TEST_CASE("project_to_halfspace_intersection: already-feasible point returned unchanged", "[solver][section-a]")
{
  Eigen::Vector3d q(1,1,1);
  Eigen::Matrix<double,Eigen::Dynamic,3> A(3,3);
  A << 1,0,0,
       0,1,0,
       0,0,1;
  Eigen::VectorXd b(3); b << 0,0,0;
  HalfspaceProjectionResult<double,3> r;
  auto status = project_to_halfspace_intersection<double,3>(q, A, b, default_options(), r);
  REQUIRE(status == HalfspaceProjectionStatus::SUCCESS);
  REQUIRE(r.p.isApprox(q));
  REQUIRE(r.active_count == 0);
}

TEST_CASE("project_to_halfspace_intersection: redundant duplicate plane", "[solver][section-a]")
{
  Eigen::Vector3d q(0,0,-5);
  Eigen::Matrix<double,Eigen::Dynamic,3> A(3,3);
  A << 0,0,1,
       0,0,1,
       0,0,1;
  Eigen::VectorXd b(3); b << 0,0,0;
  HalfspaceProjectionResult<double,3> r;
  auto status = project_to_halfspace_intersection<double,3>(q, A, b, default_options(), r);
  REQUIRE(status == HalfspaceProjectionStatus::SUCCESS);
  REQUIRE(r.p.isApprox(Eigen::Vector3d(0,0,0)));
}

TEST_CASE("project_to_halfspace_intersection: parallel consistent planes", "[solver][section-a]")
{
  Eigen::Vector3d q(0,0,-5);
  Eigen::Matrix<double,Eigen::Dynamic,3> A(2,3);
  A << 0,0,1,
       0,0,1;
  Eigen::VectorXd b(2); b << 0,-1;
  HalfspaceProjectionResult<double,3> r;
  auto status = project_to_halfspace_intersection<double,3>(q, A, b, default_options(), r);
  REQUIRE(status == HalfspaceProjectionStatus::SUCCESS);
  REQUIRE(r.p.isApprox(Eigen::Vector3d(0,0,0)));
}

TEST_CASE("project_to_halfspace_intersection: parallel contradictory planes infeasible", "[solver][section-a]")
{
  Eigen::Vector3d q(0,0,0);
  Eigen::Matrix<double,Eigen::Dynamic,3> A(2,3);
  A << 0,0,1,
       0,0,-1;
  Eigen::VectorXd b(2); b << 1,1;
  HalfspaceProjectionResult<double,3> r;
  auto status = project_to_halfspace_intersection<double,3>(q, A, b, default_options(), r);
  REQUIRE(status == HalfspaceProjectionStatus::INFEASIBLE);
}

TEST_CASE("project_to_halfspace_intersection: wedge", "[solver][section-a]")
{
  Eigen::Vector3d q(-5,-1,3);
  Eigen::Matrix<double,Eigen::Dynamic,3> A(2,3);
  A << 1,0,0,
       1,1,0;
  Eigen::VectorXd b(2); b << 0,0;
  HalfspaceProjectionResult<double,3> r;
  auto status = project_to_halfspace_intersection<double,3>(q, A, b, default_options(), r);
  REQUIRE(status == HalfspaceProjectionStatus::SUCCESS);
  REQUIRE(r.p.isApprox(Eigen::Vector3d(0,0,3)));
}

TEST_CASE("project_to_halfspace_intersection: slab", "[solver][section-a]")
{
  Eigen::Vector3d q(5,0,0);
  Eigen::Matrix<double,Eigen::Dynamic,3> A(2,3);
  A << 1,0,0,
       -1,0,0;
  Eigen::VectorXd b(2); b << 0,-1;
  HalfspaceProjectionResult<double,3> r;
  auto status = project_to_halfspace_intersection<double,3>(q, A, b, default_options(), r);
  REQUIRE(status == HalfspaceProjectionStatus::SUCCESS);
  REQUIRE(r.p.isApprox(Eigen::Vector3d(1,0,0)));
}

TEST_CASE("project_to_halfspace_intersection: bounded box", "[solver][section-a]")
{
  Eigen::Vector3d q(5,5,5);
  Eigen::Matrix<double,Eigen::Dynamic,3> A(6,3);
  A <<  1,0,0,
        0,1,0,
        0,0,1,
       -1,0,0,
        0,-1,0,
        0,0,-1;
  Eigen::VectorXd b(6); b << 0,0,0,-1,-1,-1;
  HalfspaceProjectionResult<double,3> r;
  auto status = project_to_halfspace_intersection<double,3>(q, A, b, default_options(), r);
  REQUIRE(status == HalfspaceProjectionStatus::SUCCESS);
  REQUIRE(r.p.isApprox(Eigen::Vector3d(1,1,1)));
  REQUIRE(r.active_count == 3);
}

TEST_CASE("project_to_halfspace_intersection: cone-like rank-2 active", "[solver][section-a]")
{
  Eigen::Vector3d q(-10,-2,0);
  Eigen::Matrix<double,Eigen::Dynamic,3> A(3,3);
  A << 1,1,0,
       1,-1,0,
       0,0,1;
  Eigen::VectorXd b(3); b << 0,0,-100;
  HalfspaceProjectionResult<double,3> r;
  auto status = project_to_halfspace_intersection<double,3>(q, A, b, default_options(), r);
  REQUIRE(status == HalfspaceProjectionStatus::SUCCESS);
  // p is exactly the zero vector, so isApprox's relative tolerance
  // degenerates (min-norm reference is 0); compare via absolute norm instead.
  REQUIRE(r.p.norm() < 1e-9);
  REQUIRE(r.active_count == 2);
}

TEST_CASE("project_to_halfspace_intersection: rank-1 active with redundant parallels", "[solver][section-a]")
{
  Eigen::Vector3d q(0,0,-5);
  Eigen::Matrix<double,Eigen::Dynamic,3> A(5,3);
  A << 0,0,1,
       0,0,2,
       0,0,1,
       0,0,1,
       0,0,1;
  Eigen::VectorXd b(5); b << 0,0,-3,-2,-1;
  HalfspaceProjectionResult<double,3> r;
  auto status = project_to_halfspace_intersection<double,3>(q, A, b, default_options(), r);
  REQUIRE(status == HalfspaceProjectionStatus::SUCCESS);
  REQUIRE(r.p.isApprox(Eigen::Vector3d(0,0,0)));
  REQUIRE(r.active_count == 1);
}

TEST_CASE("project_to_halfspace_intersection: near-parallel planes resolve", "[solver][section-a]")
{
  Eigen::Vector3d q(-5,0,0);
  Eigen::Matrix<double,Eigen::Dynamic,3> A(2,3);
  A << 1,0,0,
       1,1e-6,0;
  Eigen::VectorXd b(2); b << 0,0;
  HalfspaceProjectionResult<double,3> r;
  auto status = project_to_halfspace_intersection<double,3>(q, A, b, default_options(), r);
  REQUIRE(status == HalfspaceProjectionStatus::SUCCESS);
  REQUIRE(r.max_violation <= 1e-9);
}

TEST_CASE("project_to_halfspace_intersection: deterministic across repeated calls", "[solver][section-a]")
{
  Eigen::Vector3d q(3,-1,2);
  Eigen::Matrix<double,Eigen::Dynamic,3> A(4,3);
  A << 1,0,0,
       0,1,0,
       0,0,1,
       1,1,1;
  Eigen::VectorXd b(4); b << 0,0,0,-10;
  auto options = default_options();
  HalfspaceProjectionResult<double,3> r1, r2;
  project_to_halfspace_intersection<double,3>(q, A, b, options, r1);
  project_to_halfspace_intersection<double,3>(q, A, b, options, r2);
  REQUIRE(r1.p.isApprox(r2.p));
  REQUIRE(r1.active_count == r2.active_count);
}
