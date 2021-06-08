#include "../internal/monolish_internal.hpp"
#include "../internal/mpi/mpi_util.hpp"

namespace monolish {
namespace {
// std::vectror
template <typename T>
void Scatter_core(std::vector<T> &sendvec, std::vector<T> recvvec, int root,
                  MPI_Comm comm, bool gpu_sync) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

#if defined MONOLISH_USE_MPI
  MPI_Scatter(sendvec.data(), sendvec.size(),
              internal::mpi::get_type(sendvec[0]), recvvec.data(),
              recvvec.size(), internal::mpi::get_type(recvvec[0]), root, comm);
#endif

  logger.util_out();
}

// monolish::vectror
template <typename T>
void Scatter_core(monolish::vector<T> &sendvec, monolish::vector<T> recvvec,
                  int root, MPI_Comm comm, bool gpu_sync) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

#if defined MONOLISH_USE_GPU
  if ( (gpu_sync == true) && (vec.get_device_stat()==true) ) {
    sendvec.recv();
  }
#endif

#if defined MONOLISH_USE_MPI
  MPI_Scatter(sendvec.data(), sendvec.size(),
              internal::mpi::get_type(sendvec[0]), recvvec.data(),
              recvvec.size(), internal::mpi::get_type(recvvec[0]), root, comm);
#endif

#if defined MONOLISH_USE_GPU
  if (gpu_sync == true) {
    recvvec.send();
  }
#endif

  logger.util_out();
}

} // namespace
} // namespace monolish
