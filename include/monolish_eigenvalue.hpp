#pragma once
#include <omp.h>
#include <vector>

#if defined USE_MPI
#include <mpi.h>
#endif

#include "common/monolish_common.hpp"

namespace monolish {
namespace eigenvalue {

template <typename T>
int
monolish_LOBPCG(monolish::matrix::CRS<T> const &A, T& lambda, monolish::vector<T> &w);

} // namespace eigenvalue
} // namespace monolish
