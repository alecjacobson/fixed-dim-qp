// This file is part of libigl, a simple c++ geometry processing library.
//
// Copyright (C) 2026 Alec Jacobson <alecjacobson@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.
#ifndef IGL_HALFSPACE_PROJECTION_FIXTURE_H
#define IGL_HALFSPACE_PROJECTION_FIXTURE_H

#include "igl_inline.h"
#include "halfspace_projection_types.h"
#include <Eigen/Core>
#include <istream>
#include <ostream>

namespace igl
{
  /// A serializable instance of the halfspace-projection problem
  ///
  ///   min_p ||p-q||^2  s.t.  A p >= b
  ///
  /// used as the common fixture format for unit tests, differential tests
  /// against the exhaustive oracle, and failure-replay dumps.
  ///
  /// @tparam Scalar  e.g. double
  /// @tparam Dim     fixed ambient dimension (3 for the primary use case)
  template <typename Scalar, int Dim>
  struct HalfspaceProblem
  {
    Eigen::Matrix<Scalar,Dim,1> q = Eigen::Matrix<Scalar,Dim,1>::Zero();
    /// Row i is a_i (so a_i^T p >= b_i); #rows may be 0.
    Eigen::Matrix<Scalar,Eigen::Dynamic,Dim> A;
    Eigen::Matrix<Scalar,Eigen::Dynamic,1> b;
    HalfspaceProjectionOptions<Scalar> options;
  };

  /// Write a HalfspaceProblem to a plain-text stream.
  ///
  /// @param[in] out  output stream
  /// @param[in] problem  fixture to serialize
  /// @return true on success
  template <typename Scalar, int Dim>
  IGL_INLINE bool write_halfspace_projection_fixture(
    std::ostream & out,
    const HalfspaceProblem<Scalar,Dim> & problem);

  /// Read a HalfspaceProblem previously written by
  /// write_halfspace_projection_fixture(). The stream's `dim` header must
  /// match the template parameter Dim.
  ///
  /// @param[in] in  input stream
  /// @param[out] problem  parsed fixture
  /// @return true on success (false on malformed input or dim mismatch)
  template <typename Scalar, int Dim>
  IGL_INLINE bool read_halfspace_projection_fixture(
    std::istream & in,
    HalfspaceProblem<Scalar,Dim> & problem);
}

#ifndef IGL_STATIC_LIBRARY
#  include "halfspace_projection_fixture.cpp"
#endif

#endif
