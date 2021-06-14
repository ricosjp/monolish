#include "../internal/monolish_internal.hpp"
#include "../internal/mpi/mpi_util.hpp"

namespace monolish {
namespace {
// Scalar
template <typename T> void Bcast_core(T &val, int root, MPI_Comm comm) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

#if defined MONOLISH_USE_MPI
  MPI_Bcast(&val, 1, internal::mpi::get_type(val), root, comm);
#endif

  logger.util_out();
}

// std::vectror
template <typename T>
void Bcast_core(std::vector<T> &vec, int root, MPI_Comm comm) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

#if defined MONOLISH_USE_MPI
  MPI_Bcast(vec.data(), vec.size(), internal::mpi::get_type(vec.data()[0]),
            root, comm);
#endif

  logger.util_out();
}

// monolish::vectror
template <typename T>
void Bcast_core(monolish::vector<T> &vec, int root, MPI_Comm comm) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

#if defined MONOLISH_USE_MPI
  MPI_Bcast(vec.data(), vec.size(), internal::mpi::get_type(vec.data()[0]),
            root, comm);
#endif

  logger.util_out();
}

} // namespace
} // namespace monolish
