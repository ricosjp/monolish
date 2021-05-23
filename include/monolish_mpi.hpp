#pragma once

#include "common/monolish_common.hpp"
#include <climits>

#if defined MONOLISH_USE_MPI
#include <mpi.h>
#else
// MPI dammy
typedef struct ompi_communicator_t *MPI_Comm;
#endif

#if SIZE_MAX == ULONG_MAX
#define MPI_SIZE_T MPI_UNSIGNED_LONG
#elif SIZE_MAX == ULLONG_MAX
#define MPI_SIZE_T MPI_UNSIGNED_LONG_LONG
#elif SIZE_MAX == UINT_MAX
#define MPI_SIZE_T MPI_UNSIGNED
#endif

namespace monolish {
namespace mpi {
/**
 * @brief MPI class (singleton)
 */
class Comm {
private:
  MPI_Comm comm;

  Comm(){};

  Comm(MPI_Comm external_comm) { comm = external_comm; }

  ~Comm(){};

public:
  Comm(const Comm &) = delete;
  Comm &operator=(const Comm &) = delete;
  Comm(Comm &&) = delete;
  Comm &operator=(Comm &&) = delete;

  static Comm &get_instance() {
    static Comm instance;
    return instance;
  }

  MPI_Comm get_comm() { return comm; }

  void Init();
  void Init(int argc, char **argv);
  bool Initialized();
  void Finalize();

#include "mpi/allreduce.hpp"
};
} // namespace mpi
} // namespace monolish
