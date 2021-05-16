#pragma once

#if defined MONOLISH_USE_MPI
#include <mpi.h>
#else
// MPI dammy
typedef struct ompi_communicator_t *MPI_Comm;
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
  size_t LogLevel = 0;

  Comm(const Comm &) = delete;
  Comm &operator=(const Comm &) = delete;
  Comm(Comm &&) = delete;
  Comm &operator=(Comm &&) = delete;

  static Comm &get_instance() {
    static Comm instance;
    return instance;
  }

  void Init();
  void Init(int argc, char **argv);

  void Finalize();
};
} // namespace mpi
} // namespace monolish
