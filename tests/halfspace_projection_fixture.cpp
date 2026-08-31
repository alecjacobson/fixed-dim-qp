#include <catch2/catch_test_macros.hpp>
#include <igl/halfspace_projection_fixture.h>
#include <sstream>

TEST_CASE("halfspace_projection_fixture: round trip", "[fixture]")
{
  igl::HalfspaceProblem<double,3> problem;
  problem.q << 1.5, -2.25, 0.125;
  problem.A.resize(2,3);
  problem.A << 1,0,0,
               0,1,0;
  problem.b.resize(2);
  problem.b << 0.5, -0.5;
  problem.options.eps_rank = 1e-8;
  problem.options.eps_feasible = 1e-7;
  problem.options.eps_kkt = 1e-6;
  problem.options.eps_outward = 1e-3;
  problem.options.seed = 12345;
  problem.options.max_planes = 64;

  std::stringstream ss;
  REQUIRE(igl::write_halfspace_projection_fixture(ss, problem));

  igl::HalfspaceProblem<double,3> parsed;
  REQUIRE(igl::read_halfspace_projection_fixture(ss, parsed));

  REQUIRE(parsed.q.isApprox(problem.q));
  REQUIRE(parsed.A.isApprox(problem.A));
  REQUIRE(parsed.b.isApprox(problem.b));
  REQUIRE(parsed.options.eps_rank == problem.options.eps_rank);
  REQUIRE(parsed.options.eps_feasible == problem.options.eps_feasible);
  REQUIRE(parsed.options.eps_kkt == problem.options.eps_kkt);
  REQUIRE(parsed.options.eps_outward == problem.options.eps_outward);
  REQUIRE(parsed.options.seed == problem.options.seed);
  REQUIRE(parsed.options.max_planes == problem.options.max_planes);
}

TEST_CASE("halfspace_projection_fixture: no planes", "[fixture]")
{
  igl::HalfspaceProblem<double,3> problem;
  problem.q << 0,0,0;
  problem.A.resize(0,3);
  problem.b.resize(0);

  std::stringstream ss;
  REQUIRE(igl::write_halfspace_projection_fixture(ss, problem));
  igl::HalfspaceProblem<double,3> parsed;
  REQUIRE(igl::read_halfspace_projection_fixture(ss, parsed));
  REQUIRE(parsed.A.rows() == 0);
  REQUIRE(parsed.b.rows() == 0);
}

TEST_CASE("halfspace_projection_fixture: dim mismatch rejected", "[fixture]")
{
  igl::HalfspaceProblem<double,3> problem;
  problem.q << 0,0,0;
  problem.A.resize(0,3);
  problem.b.resize(0);
  std::stringstream ss;
  REQUIRE(igl::write_halfspace_projection_fixture(ss, problem));

  igl::HalfspaceProblem<double,2> parsed;
  REQUIRE_FALSE(igl::read_halfspace_projection_fixture(ss, parsed));
}
