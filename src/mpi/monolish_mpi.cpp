#include "../internal/monolish_internal.hpp"

namespace monolish {
namespace mpi {

void Comm::Init() {
#if defined MONOLISH_USE_MPI
  MPI_Init(nullptr, nullptr);
#endif
}

void Comm::Init(int argc, char **argv) {
#if defined MONOLISH_USE_MPI
  MPI_Init(&argc, &argv);
#endif
}

void Comm::Finalize() {
#if defined MONOLISH_USE_MPI
  MPI_Finalize();
#endif
}
} // namespace mpi
} // namespace monolish
