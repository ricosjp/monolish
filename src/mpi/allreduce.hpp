#include "../internal/monolish_internal.hpp"
#include "../internal/mpi/mpi_util.hpp"

namespace monolish {
namespace {
template <typename T> T Allreduce_core(T val, MPI_Op op, MPI_Comm comm) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

#if defined MONOLISH_USE_MPI
  MPI_Allreduce(&val, &val, 1, internal::mpi::get_type(val), op, comm);
#endif

  logger.util_out();
  return val;
}
} // namespace
} // namespace monolish
