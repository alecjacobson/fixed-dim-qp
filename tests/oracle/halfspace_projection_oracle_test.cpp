#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "halfspace_projection_oracle.h"

using namespace fixed_dim_qp_oracle;
using igl::HalfspaceProjectionOptions;
using igl::HalfspaceProjectionStatus;
using Catch::Approx;

namespace
{
  HalfspaceProjectionOptions<double> default_options()
  {
    HalfspaceProjectionOptions<double> o;
    o.eps_rank = 1e-9;
    o.eps_feasible = 1e-9;
    o.eps_kkt = 1e-9;
    o.eps_outward = 0;
    return o;
  }
}

TEST_CASE("oracle: no planes returns q", "[oracle][section-a]")
{
  Eigen::Vector3d q(1,2,3);
  Eigen::Matrix<double,Eigen::Dynamic,3> A(0,3);
  Eigen::VectorXd b(0);
  auto r = oracle_project_to_halfspace_intersection<double,3>(q, A, b, default_options());
  REQUIRE(r.status == HalfspaceProjectionStatus::SUCCESS);
  REQUIRE(r.p.isApprox(q));
  REQUIRE(r.active_ids.empty());
}

TEST_CASE("oracle: one plane is ordinary point-to-plane projection", "[oracle][section-a]")
{
  // Plane z >= 0, q below it.
  Eigen::Vector3d q(0,0,-5);
  Eigen::Matrix<double,Eigen::Dynamic,3> A(1,3);
  A << 0,0,1;
  Eigen::VectorXd b(1); b << 0;
  auto r = oracle_project_to_halfspace_intersection<double,3>(q, A, b, default_options());
  REQUIRE(r.status == HalfspaceProjectionStatus::SUCCESS);
  REQUIRE(r.p.isApprox(Eigen::Vector3d(0,0,0)));
  REQUIRE(r.active_ids.size() == 1);
}

TEST_CASE("oracle: two orthogonal planes project to their intersection line", "[oracle][section-a]")
{
  // x >= 0, y >= 0; q = (-3,-4,7) -> p = (0,0,7)
  Eigen::Vector3d q(-3,-4,7);
  Eigen::Matrix<double,Eigen::Dynamic,3> A(2,3);
  A << 1,0,0,
       0,1,0;
  Eigen::VectorXd b(2); b << 0,0;
  auto r = oracle_project_to_halfspace_intersection<double,3>(q, A, b, default_options());
  REQUIRE(r.status == HalfspaceProjectionStatus::SUCCESS);
  REQUIRE(r.p.isApprox(Eigen::Vector3d(0,0,7)));
  REQUIRE(r.active_ids.size() == 2);
}

TEST_CASE("oracle: three orthogonal planes project to their intersection point", "[oracle][section-a]")
{
  // x>=0,y>=0,z>=0; q=(-1,-2,-3) -> p=(0,0,0)
  Eigen::Vector3d q(-1,-2,-3);
  Eigen::Matrix<double,Eigen::Dynamic,3> A(3,3);
  A << 1,0,0,
       0,1,0,
       0,0,1;
  Eigen::VectorXd b(3); b << 0,0,0;
  auto r = oracle_project_to_halfspace_intersection<double,3>(q, A, b, default_options());
  REQUIRE(r.status == HalfspaceProjectionStatus::SUCCESS);
  REQUIRE(r.p.isApprox(Eigen::Vector3d(0,0,0)));
  REQUIRE(r.active_ids.size() == 3);
}

TEST_CASE("oracle: preferred point already feasible is returned unchanged", "[oracle][section-a]")
{
  Eigen::Vector3d q(1,1,1);
  Eigen::Matrix<double,Eigen::Dynamic,3> A(3,3);
  A << 1,0,0,
       0,1,0,
       0,0,1;
  Eigen::VectorXd b(3); b << 0,0,0;
  auto r = oracle_project_to_halfspace_intersection<double,3>(q, A, b, default_options());
  REQUIRE(r.status == HalfspaceProjectionStatus::SUCCESS);
  REQUIRE(r.p.isApprox(q));
  REQUIRE(r.active_ids.empty());
}

TEST_CASE("oracle: redundant duplicate plane does not change result", "[oracle][section-a]")
{
  Eigen::Vector3d q(0,0,-5);
  Eigen::Matrix<double,Eigen::Dynamic,3> A(3,3);
  A << 0,0,1,
       0,0,1,
       0,0,1;
  Eigen::VectorXd b(3); b << 0,0,0;
  auto r = oracle_project_to_halfspace_intersection<double,3>(q, A, b, default_options());
  REQUIRE(r.status == HalfspaceProjectionStatus::SUCCESS);
  REQUIRE(r.p.isApprox(Eigen::Vector3d(0,0,0)));
}

TEST_CASE("oracle: parallel consistent planes", "[oracle][section-a]")
{
  // z >= 0 and z >= -1: binding one is z>=0.
  Eigen::Vector3d q(0,0,-5);
  Eigen::Matrix<double,Eigen::Dynamic,3> A(2,3);
  A << 0,0,1,
       0,0,1;
  Eigen::VectorXd b(2); b << 0, -1;
  auto r = oracle_project_to_halfspace_intersection<double,3>(q, A, b, default_options());
  REQUIRE(r.status == HalfspaceProjectionStatus::SUCCESS);
  REQUIRE(r.p.isApprox(Eigen::Vector3d(0,0,0)));
}

TEST_CASE("oracle: parallel contradictory planes are infeasible", "[oracle][section-a]")
{
  // z >= 1 and -z >= 1 (i.e. z <= -1): empty.
  Eigen::Vector3d q(0,0,0);
  Eigen::Matrix<double,Eigen::Dynamic,3> A(2,3);
  A << 0,0,1,
       0,0,-1;
  Eigen::VectorXd b(2); b << 1, 1;
  auto r = oracle_project_to_halfspace_intersection<double,3>(q, A, b, default_options());
  REQUIRE(r.status == HalfspaceProjectionStatus::INFEASIBLE);
}

TEST_CASE("oracle: wedge (two non-orthogonal planes)", "[oracle][section-a]")
{
  // x>=0 and x+y>=0 (wedge along z-axis); q off the bisector so both
  // constraints are strictly active (nonzero multipliers).
  Eigen::Vector3d q(-5,-1,3);
  Eigen::Matrix<double,Eigen::Dynamic,3> A(2,3);
  A << 1,0,0,
       1,1,0;
  Eigen::VectorXd b(2); b << 0,0;
  auto r = oracle_project_to_halfspace_intersection<double,3>(q, A, b, default_options());
  REQUIRE(r.status == HalfspaceProjectionStatus::SUCCESS);
  REQUIRE(r.p.isApprox(Eigen::Vector3d(0,0,3)));
  REQUIRE(r.active_ids.size() == 2);
}

TEST_CASE("oracle: slab (two opposite-facing planes)", "[oracle][section-a]")
{
  // 0 <= x <= 1 as two halfspaces: x>=0, -x>=-1; q=(5,0,0) -> p=(1,0,0)
  Eigen::Vector3d q(5,0,0);
  Eigen::Matrix<double,Eigen::Dynamic,3> A(2,3);
  A << 1,0,0,
       -1,0,0;
  Eigen::VectorXd b(2); b << 0, -1;
  auto r = oracle_project_to_halfspace_intersection<double,3>(q, A, b, default_options());
  REQUIRE(r.status == HalfspaceProjectionStatus::SUCCESS);
  REQUIRE(r.p.isApprox(Eigen::Vector3d(1,0,0)));
}

TEST_CASE("oracle: bounded box", "[oracle][section-a]")
{
  // unit box [0,1]^3, q way outside -> nearest corner (1,1,1)
  Eigen::Vector3d q(5,5,5);
  Eigen::Matrix<double,Eigen::Dynamic,3> A(6,3);
  A <<  1,0,0,
        0,1,0,
        0,0,1,
       -1,0,0,
        0,-1,0,
        0,0,-1;
  Eigen::VectorXd b(6); b << 0,0,0,-1,-1,-1;
  auto r = oracle_project_to_halfspace_intersection<double,3>(q, A, b, default_options());
  REQUIRE(r.status == HalfspaceProjectionStatus::SUCCESS);
  REQUIRE(r.p.isApprox(Eigen::Vector3d(1,1,1)));
  REQUIRE(r.active_ids.size() == 3);
}

TEST_CASE("oracle: cone-like configuration (rank-2 active solution)", "[oracle][section-a]")
{
  // Two planes whose intersection is a line not aligned with an axis, third
  // far away; q off the bisector so both constraints are strictly active.
  Eigen::Vector3d q(-10,-2,0);
  Eigen::Matrix<double,Eigen::Dynamic,3> A(3,3);
  A << 1,1,0,   // x+y>=0
       1,-1,0,  // x-y>=0
       0,0,1;   // z >= -100 (inactive)
  Eigen::VectorXd b(3); b << 0,0,-100;
  auto r = oracle_project_to_halfspace_intersection<double,3>(q, A, b, default_options());
  REQUIRE(r.status == HalfspaceProjectionStatus::SUCCESS);
  REQUIRE(r.p.isApprox(Eigen::Vector3d(0,0,0)));
  REQUIRE(r.active_ids.size() == 2);
}

TEST_CASE("oracle: rank-1 active solution with many redundant parallel planes", "[oracle][section-a]")
{
  Eigen::Vector3d q(0,0,-5);
  Eigen::Matrix<double,Eigen::Dynamic,3> A(5,3);
  A << 0,0,1,
       0,0,2,   // same direction, unnormalized
       0,0,1,
       0,0,1,
       0,0,1;
  Eigen::VectorXd b(5); b << 0,0,-3,-2,-1;
  auto r = oracle_project_to_halfspace_intersection<double,3>(q, A, b, default_options());
  REQUIRE(r.status == HalfspaceProjectionStatus::SUCCESS);
  REQUIRE(r.p.isApprox(Eigen::Vector3d(0,0,0)));
  REQUIRE(r.active_ids.size() == 1);
}

TEST_CASE("oracle: near-parallel planes still resolve correctly", "[oracle][section-a]")
{
  Eigen::Vector3d q(-5,0,0);
  Eigen::Matrix<double,Eigen::Dynamic,3> A(2,3);
  A << 1,0,0,
       1,1e-6,0;
  Eigen::VectorXd b(2); b << 0,0;
  auto r = oracle_project_to_halfspace_intersection<double,3>(q, A, b, default_options());
  REQUIRE(r.status == HalfspaceProjectionStatus::SUCCESS);
  REQUIRE(r.max_violation <= 1e-9);
}
