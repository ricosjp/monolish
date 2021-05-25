#pragma once

#include "common/monolish_common.hpp"
#include <climits>

#if defined MONOLISH_USE_MPI
#include <mpi.h>
#else
// MPI dummy
#include "mpi/mpi_dummy.hpp"
#endif

#if SIZE_MAX == ULONG_MAX
#define MPI_SIZE_T MPI_UNSIGNED_LONG
#elif SIZE_MAX == ULLONG_MAX
#define MPI_SIZE_T MPI_UNSIGNED_LONG_LONG
#elif SIZE_MAX == UINT_MAX
#define MPI_SIZE_T MPI_UNSIGNED
#endif

namespace monolish {
/**
 * @brief
 * C++ template MPI class, Functions of this class do nothing when MPI is
 * disabled.
 * Functions in this class are under development. Many BLAS functions don't
 * support MPI.
 */
namespace mpi {
/**
 * @brief MPI class (singleton)
 */
class Comm {
private:
  /**
   * @brief MPI communicator, MPI_COMM_WORLD
   */
  MPI_Comm comm;
  int rank;
  int size;

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

  /**
   * @brief get my rank number
   * @return rank number
   */
  [[nodiscard]] int get_rank() { return rank; }

  /**
   * @brief get the number of processes
   * @return the number of prodessed
   */
  [[nodiscard]] int get_size() { return size; }

  /**
   * @brief get communicator
   * @return MPI_COMM_WORLD
   */
  [[nodiscard]] MPI_Comm get_comm() { return comm; }

  /**
   * @brief Initialize the MPI execution environment
   */
  void Init();

  /**
   * @brief Initialize the MPI execution environment
   * @param argc Pointer to the number of arguments
   * @param argv Pointer to the argument vector
   * */
  void Init(int argc, char **argv);

  /**
   * @brief Indicates whether MPI_Init has been called
   * @return true: initialized, false: not initialized
   * */
  bool Initialized();

  /**
   * @brief Terminates MPI execution environment
   * */
  void Finalize();

#include "mpi/allreduce.hpp"
};
} // namespace mpi
} // namespace monolish
