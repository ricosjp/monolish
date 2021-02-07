#include "../../../include/monolish_vml.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {
  namespace{
    template <typename F1, typename F2, typename F3>
      void mmpow_core(const F1 &A, const F2 &B, F3 &C) {
        Logger &logger = Logger::get_instance();
        logger.func_in(monolish_func);

        // err
        assert(util::is_same_size(A, B, C));
        assert(util::is_same_structure(A, B, C));
        assert(util::is_same_device_mem_stat(A, B, C));

        internal::vpow(A.get_nnz(), A.val.data(), B.val.data(), C.val.data(),
            C.get_device_mem_stat());

        logger.func_out();
      }

    template <typename F1, typename F2, typename F3>
      void smpow_core(const F1 &A, const F2 &alpha, F3 &C) {
        Logger &logger = Logger::get_instance();
        logger.func_in(monolish_func);

        // err
        assert(util::is_same_size(A, C));
        assert(util::is_same_structure(A, C));
        assert(util::is_same_device_mem_stat(A, C));

        internal::vpow(A.get_nnz(), A.val.data(), alpha, C.val.data(),
            C.get_device_mem_stat());

        logger.func_out();
      }
  }

  namespace vml {
    void pow(const matrix::Dense<double> &A, const matrix::Dense<double> &B, matrix::Dense<double> &C) {
      mmpow_core(A, B, C);
    }
    void pow(const matrix::Dense<float> &A, const matrix::Dense<float> &B, matrix::Dense<float> &C) {
      mmpow_core(A, B, C);
    }

    void pow(const matrix::CRS<double> &A, const matrix::CRS<double> &B, matrix::CRS<double> &C) {
      mmpow_core(A, B, C);
    }
    void pow(const matrix::CRS<float> &A, const matrix::CRS<float> &B, matrix::CRS<float> &C) {
      mmpow_core(A, B, C);
    }


    void pow(const matrix::Dense<double> &A, const double alpha, matrix::Dense<double> &C) {
      smpow_core(A, alpha, C);
    }
    void pow(const matrix::Dense<float> &A, const float alpha, matrix::Dense<float> &C) {
      smpow_core(A, alpha, C);
    }
    void pow(const matrix::CRS<double> &A, const double alpha, matrix::CRS<double> &C) {
      smpow_core(A, alpha, C);
    }
    void pow(const matrix::CRS<float> &A, const float alpha, matrix::CRS<float> &C) {
      smpow_core(A, alpha, C);
    }

  }
}
