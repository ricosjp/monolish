#pragma once

#if defined MONOLISH_USE_MPI
#include <mpi.h>
#endif

namespace monolish {
  namespace MPI {
    /**
     * @brief MPI class (singleton)
     */
    class Comm {
      private:

        Comm() = default;
        ~Comm() {};

        MPI_Comm comm;

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

    };
  } // namespace monolish
}
