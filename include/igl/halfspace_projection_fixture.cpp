#include "halfspace_projection_fixture.h"
#include <string>
#include <vector>

template <typename Scalar, int Dim>
IGL_INLINE bool igl::write_halfspace_projection_fixture(
  std::ostream & out,
  const igl::HalfspaceProblem<Scalar,Dim> & problem)
{
  out.precision(17);
  out << "FIXED_DIM_QP_FIXTURE 1\n";
  out << "dim " << Dim << "\n";
  out << "q";
  for(int j = 0;j < Dim;j++) out << " " << problem.q(j);
  out << "\n";
  out << "options"
      << " eps_rank " << problem.options.eps_rank
      << " eps_feasible " << problem.options.eps_feasible
      << " eps_kkt " << problem.options.eps_kkt
      << " eps_outward " << problem.options.eps_outward
      << " seed " << problem.options.seed
      << " max_planes " << problem.options.max_planes
      << "\n";
  const Eigen::Index m = problem.A.rows();
  out << "planes " << m << "\n";
  for(Eigen::Index i = 0;i < m;i++)
  {
    for(int j = 0;j < Dim;j++) out << problem.A(i,j) << " ";
    out << problem.b(i) << "\n";
  }
  return static_cast<bool>(out);
}

template <typename Scalar, int Dim>
IGL_INLINE bool igl::read_halfspace_projection_fixture(
  std::istream & in,
  igl::HalfspaceProblem<Scalar,Dim> & problem)
{
  std::string tag;
  int version = 0;
  if(!(in >> tag >> version) || tag != "FIXED_DIM_QP_FIXTURE") return false;

  std::string key;
  int file_dim = -1;
  if(!(in >> key >> file_dim) || key != "dim") return false;
  if(file_dim != Dim) return false;

  if(!(in >> key) || key != "q") return false;
  for(int j = 0;j < Dim;j++)
  {
    if(!(in >> problem.q(j))) return false;
  }

  if(!(in >> key) || key != "options") return false;
  std::string field;
  auto expect_field = [&](const char * name, Scalar & dst)->bool
  {
    if(!(in >> field) || field != name) return false;
    return static_cast<bool>(in >> dst);
  };
  if(!expect_field("eps_rank", problem.options.eps_rank)) return false;
  if(!expect_field("eps_feasible", problem.options.eps_feasible)) return false;
  if(!expect_field("eps_kkt", problem.options.eps_kkt)) return false;
  if(!expect_field("eps_outward", problem.options.eps_outward)) return false;
  if(!(in >> field) || field != "seed") return false;
  if(!(in >> problem.options.seed)) return false;
  if(!(in >> field) || field != "max_planes") return false;
  if(!(in >> problem.options.max_planes)) return false;

  Eigen::Index m = 0;
  if(!(in >> key >> m) || key != "planes") return false;
  problem.A.resize(m,Dim);
  problem.b.resize(m);
  for(Eigen::Index i = 0;i < m;i++)
  {
    for(int j = 0;j < Dim;j++)
    {
      if(!(in >> problem.A(i,j))) return false;
    }
    if(!(in >> problem.b(i))) return false;
  }
  return true;
}

#ifdef IGL_STATIC_LIBRARY
template bool igl::write_halfspace_projection_fixture<double,3>(std::ostream &, const igl::HalfspaceProblem<double,3> &);
template bool igl::read_halfspace_projection_fixture<double,3>(std::istream &, igl::HalfspaceProblem<double,3> &);
#endif
