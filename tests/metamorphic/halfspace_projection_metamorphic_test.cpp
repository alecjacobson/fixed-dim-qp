// Section C metamorphic tests. Deliberately built on a handful of
// hand-constructed base cases (not the randomized generator) so failures are
// unambiguous and the tests are not flaky under rare degenerate draws.
// See progressive_hulls_fixed_dim_qp_design.md, "Metamorphic tests".
#include <catch2/catch_test_macros.hpp>
#include <igl/project_to_halfspace_intersection.h>
#include <Eigen/Geometry>
#include <vector>
#include <string>
#include <limits>
#include <cmath>

using igl::HalfspaceProjectionOptions;
using igl::HalfspaceProjectionResult;
using igl::HalfspaceProjectionStatus;

namespace
{
  struct BaseCase
  {
    std::string name;
    Eigen::Vector3d q;
    Eigen::Matrix<double,Eigen::Dynamic,3> A;
    Eigen::VectorXd b;
  };

  std::vector<BaseCase> base_cases()
  {
    std::vector<BaseCase> cases;
    {
      BaseCase c;
      c.name = "unit box";
      c.q = Eigen::Vector3d(5,5,5);
      c.A.resize(6,3);
      c.A <<  1,0,0,  0,1,0,  0,0,1,
             -1,0,0,  0,-1,0, 0,0,-1;
      c.b.resize(6);
      c.b << 0,0,0,-1,-1,-1;
      cases.push_back(c);
    }
    {
      BaseCase c;
      c.name = "wedge";
      c.q = Eigen::Vector3d(-5,-1,3);
      c.A.resize(2,3);
      c.A << 1,0,0,
             1,1,0;
      c.b.resize(2);
      c.b << 0,0;
      cases.push_back(c);
    }
    {
      BaseCase c;
      c.name = "single plane";
      c.q = Eigen::Vector3d(0,0,-5);
      c.A.resize(1,3);
      c.A << 0,0,1;
      c.b.resize(1);
      c.b << 0;
      cases.push_back(c);
    }
    return cases;
  }

  HalfspaceProjectionOptions<double> options()
  {
    HalfspaceProjectionOptions<double> o;
    o.eps_rank = 1e-9;
    o.eps_feasible = 1e-9;
    o.eps_kkt = 1e-8;
    o.eps_outward = 0;
    o.seed = 7;
    return o;
  }
}

TEST_CASE("metamorphic: adding a trivially-redundant plane does not change result", "[metamorphic][section-c]")
{
  for(const auto & c : base_cases())
  {
    HalfspaceProjectionResult<double,3> r0;
    REQUIRE(igl::project_to_halfspace_intersection<double,3>(c.q, c.A, c.b, options(), r0)
      == HalfspaceProjectionStatus::SUCCESS);

    Eigen::Matrix<double,Eigen::Dynamic,3> A2(c.A.rows() + 1, 3);
    Eigen::VectorXd b2(c.b.size() + 1);
    A2.topRows(c.A.rows()) = c.A;
    b2.head(c.b.size()) = c.b;
    A2.row(c.A.rows()) = Eigen::RowVector3d(0,0,-1);
    b2(c.b.size()) = -1e6; // z >= -1e6: always satisfied here

    HalfspaceProjectionResult<double,3> r1;
    REQUIRE(igl::project_to_halfspace_intersection<double,3>(c.q, A2, b2, options(), r1)
      == HalfspaceProjectionStatus::SUCCESS);

    INFO(c.name);
    REQUIRE(r1.p.isApprox(r0.p, 1e-8));
  }
}

TEST_CASE("metamorphic: duplicating a plane many times does not change result", "[metamorphic][section-c]")
{
  for(const auto & c : base_cases())
  {
    HalfspaceProjectionResult<double,3> r0;
    REQUIRE(igl::project_to_halfspace_intersection<double,3>(c.q, c.A, c.b, options(), r0)
      == HalfspaceProjectionStatus::SUCCESS);

    const int extra = 20;
    Eigen::Matrix<double,Eigen::Dynamic,3> A2(c.A.rows() + extra, 3);
    Eigen::VectorXd b2(c.b.size() + extra);
    A2.topRows(c.A.rows()) = c.A;
    b2.head(c.b.size()) = c.b;
    for(int k = 0;k < extra;k++)
    {
      A2.row(c.A.rows() + k) = c.A.row(0);
      b2(c.b.size() + k) = c.b(0);
    }

    HalfspaceProjectionResult<double,3> r1;
    REQUIRE(igl::project_to_halfspace_intersection<double,3>(c.q, A2, b2, options(), r1)
      == HalfspaceProjectionStatus::SUCCESS);

    INFO(c.name);
    REQUIRE(r1.p.isApprox(r0.p, 1e-8));
  }
}

TEST_CASE("metamorphic: rigid transform + uniform scale commutes with the solve", "[metamorphic][section-c]")
{
  const Eigen::Vector3d axis = Eigen::Vector3d(1,2,3).normalized();
  const Eigen::Matrix3d R = Eigen::AngleAxisd(37.0 * M_PI / 180.0, axis).toRotationMatrix();
  const Eigen::Vector3d t(3,-2,5);
  const double s = 2.5;

  for(const auto & c : base_cases())
  {
    HalfspaceProjectionResult<double,3> r0;
    REQUIRE(igl::project_to_halfspace_intersection<double,3>(c.q, c.A, c.b, options(), r0)
      == HalfspaceProjectionStatus::SUCCESS);

    const Eigen::Vector3d q2 = s * (R * c.q) + t;
    Eigen::Matrix<double,Eigen::Dynamic,3> A2(c.A.rows(), 3);
    Eigen::VectorXd b2(c.b.size());
    for(Eigen::Index i = 0;i < c.A.rows();i++)
    {
      const Eigen::Vector3d a2 = R * c.A.row(i).transpose();
      A2.row(i) = a2;
      b2(i) = s * c.b(i) + a2.dot(t);
    }

    HalfspaceProjectionResult<double,3> r1;
    REQUIRE(igl::project_to_halfspace_intersection<double,3>(q2, A2, b2, options(), r1)
      == HalfspaceProjectionStatus::SUCCESS);

    const Eigen::Vector3d expected = s * (R * r0.p) + t;
    INFO(c.name);
    REQUIRE(r1.p.isApprox(expected, 1e-6));
  }
}

TEST_CASE("metamorphic: increasing eps_outward moves residuals outward monotonically", "[metamorphic][section-c]")
{
  const std::vector<double> offsets = {0.0, 0.01, 0.05, 0.1, 0.2};

  for(const auto & c : base_cases())
  {
    double prev_max_violation = std::numeric_limits<double>::infinity();
    for(const double eps_outward : offsets)
    {
      auto o = options();
      o.eps_outward = eps_outward;
      HalfspaceProjectionResult<double,3> r;
      const auto status = igl::project_to_halfspace_intersection<double,3>(c.q, c.A, c.b, o, r);
      INFO(c.name << " eps_outward=" << eps_outward);
      REQUIRE(status == HalfspaceProjectionStatus::SUCCESS);
      // max_violation is reported against the *original* b, so as the
      // solve is pushed further inside (eps_outward growing), it must not
      // get worse.
      REQUIRE(r.max_violation <= prev_max_violation + 1e-9);
      prev_max_violation = r.max_violation;
    }
  }
}

TEST_CASE("metamorphic: large regularization approaches projection of p0", "[metamorphic][section-c]")
{
  // min g^T p + lambda||p-p0||^2 s.t. Ap>=b  <=>  project q=p0-g/(2*lambda).
  // As lambda -> infinity, q -> p0, so the solve should converge to
  // project_to_halfspace_intersection(p0, A, b, ...).
  const Eigen::Vector3d p0(5,5,5);
  const Eigen::Vector3d g(1,2,-3);
  const std::vector<double> lambdas = {0.1, 1.0, 10.0, 100.0, 1000.0, 100000.0};

  for(const auto & c : base_cases())
  {
    HalfspaceProjectionResult<double,3> r_ref;
    REQUIRE(igl::project_to_halfspace_intersection<double,3>(p0, c.A, c.b, options(), r_ref)
      == HalfspaceProjectionStatus::SUCCESS);

    double prev_error = std::numeric_limits<double>::infinity();
    for(const double lambda : lambdas)
    {
      const Eigen::Vector3d q = p0 - g / (2.0 * lambda);
      HalfspaceProjectionResult<double,3> r;
      REQUIRE(igl::project_to_halfspace_intersection<double,3>(q, c.A, c.b, options(), r)
        == HalfspaceProjectionStatus::SUCCESS);
      const double error = (r.p - r_ref.p).norm();
      INFO(c.name << " lambda=" << lambda << " error=" << error << " prev=" << prev_error);
      REQUIRE(error <= prev_error + 1e-9);
      prev_error = error;
    }
    REQUIRE(prev_error < 1e-3);
  }
}
