#include "../internal/monolish_internal.hpp"

namespace monolish {
namespace mpi {

void Comm::Init() {
#if defined MONOLISH_USE_MPI
  assert(MPI_Init(nullptr, nullptr) == MPI_SUCCESS);
#endif
}

void Comm::Init(int argc, char **argv) {
#if defined MONOLISH_USE_MPI
  assert(MPI_Init(&argc, &argv) == MPI_SUCCESS);
#endif
}

bool Comm::Initialized() {
#if defined MONOLISH_USE_MPI
  int flag;
  assert(MPI_Initialized(&flag) == MPI_SUCCESS);;
  return ( flag == 0 ) ? true : false;
#endif
}

void Comm::Finalize() {
#if defined MONOLISH_USE_MPI
  assert(MPI_Finalize() == MPI_SUCCESS);;
#endif
}

} // namespace mpi
} // namespace monolish
