#include "../../../../include/monolish_blas.hpp"
#include "../../../internal/monolish_internal.hpp"

namespace monolish {

  namespace{
    template <typename F>
    void copy_core(const matrix::Dense<F> &A, matrix::Dense<F> &C) {
      Logger &logger = Logger::get_instance();
      logger.util_in(monolish_func);

      // err
      assert(util::is_same_size(A, C));
      assert(util::is_same_device_mem_stat(A, C));

      internal::vcopy(A.get_nnz(), A.val.data(), C.val.data(),
          A.get_device_mem_stat());

      logger.util_out();
    }
  }

  namespace blas{

    void copy(const matrix::Dense<double> &A, matrix::Dense<double> &C) { copy_core(A, C);}
    void copy(const matrix::Dense<float> &A, matrix::Dense<float> &C) { copy_core(A, C);}

  }
}
