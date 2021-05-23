#include "../internal/monolish_internal.hpp"
#include<typeinfo>

namespace monolish {
  namespace {
    template <typename T> T Allreduce_core(T val, MPI_Op op, MPI_Comm comm) {
      Logger &logger = Logger::get_instance();
      logger.func_in(monolish_func);

      if(typeid(double) == typeid(val)){
        MPI_Allreduce(&val, &val, 1, MPI_DOUBLE, op, comm);
      }
      if(typeid(float) == typeid(val)){
        MPI_Allreduce(&val, &val, 1, MPI_FLOAT, op, comm);
      }
      if(typeid(int) == typeid(val)){
        MPI_Allreduce(&val, &val, 1, MPI_INT, op, comm);
      }
      if(typeid(size_t) == typeid(val)){
        MPI_Allreduce(&val, &val, 1, MPI_SIZE_T, op, comm);
      }

      logger.func_out();
      return val;
    }
  }
}
