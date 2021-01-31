#include "../../../../include/monolish_blas.hpp"
#include "../../../internal/monolish_internal.hpp"

namespace monolish {

  namespace{
// double ///////////////////
  template<typename VEC>
    void Dmatvec_core(const matrix::CRS<double> &A, const VEC &x, VEC &y) {
      Logger &logger = Logger::get_instance();
      logger.func_in(monolish_func);

      // err, M = MN * N
      assert(A.get_row() == y.size());
      assert(A.get_col() == x.size());
      assert(util::is_same_device_mem_stat(A, x, y));

      const double *vald = A.val.data();
      const double *xd = x.data();
      const int *rowd = A.row_ptr.data();
      const int *cold = A.col_ind.data();
      const int m = A.get_row();
      const int n = A.get_col();
      const int nnz = A.get_nnz();
      double *yd = y.data();

      if (A.get_device_mem_stat() == true) {
#if MONOLISH_USE_GPU // gpu
        cusparseHandle_t sp_handle;
        cusparseCreate(&sp_handle);
        cudaDeviceSynchronize();

        cusparseMatDescr_t descr = 0;
        cusparseCreateMatDescr(&descr);
        cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
        cusparseSetMatFillMode(descr, CUSPARSE_FILL_MODE_LOWER);
        cusparseSetMatDiagType(descr, CUSPARSE_DIAG_TYPE_UNIT);

        const cusparseOperation_t trans = CUSPARSE_OPERATION_NON_TRANSPOSE;

        const double alpha = 1.0;
        const double beta = 0.0;

#pragma omp target data use_device_ptr(xd, yd, vald, rowd, cold)
        {
          internal::check_CUDA(cusparseDcsrmv(sp_handle, trans, m, n, nnz, &alpha,
                descr, vald, rowd, cold, xd, &beta,
                yd));
        }
#else
        throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
      } else {
        // MKL
#if MONOLISH_USE_MKL
        const double alpha = 1.0;
        const double beta = 0.0;
        mkl_dcsrmv("N", &m, &n, &alpha, "G__C", vald, cold, rowd, rowd + 1, xd,
            &beta, yd);

        // OSS
#else
#pragma omp parallel for
        for (int i = 0; i < (int)A.get_row(); i++) {
          double ytmp = 0.0;
          for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; j++) {
            ytmp += vald[j] * xd[A.col_ind[j]];
          }
          y[i] = ytmp;
        }
#endif
      }

      logger.func_out();
    }

// float ///////////////////
  template<typename VEC>
    void Smatvec_core(const matrix::CRS<float> &A, const VEC &x, VEC &y) {
      Logger &logger = Logger::get_instance();
      logger.func_in(monolish_func);

      // err, M = MN * N
      assert(A.get_row() == y.size());
      assert(A.get_col() == x.size());
      assert(util::is_same_device_mem_stat(A, x, y));

      const float *vald = A.val.data();
      const float *xd = x.data();
      const int *rowd = A.row_ptr.data();
      const int *cold = A.col_ind.data();
      const int m = A.get_row();
      const int n = A.get_col();
      const int nnz = A.get_nnz();
      float *yd = y.data();

      if (A.get_device_mem_stat() == true) {
#if MONOLISH_USE_GPU // gpu
        cusparseHandle_t sp_handle;
        cusparseCreate(&sp_handle);
        cudaDeviceSynchronize();

        cusparseMatDescr_t descr = 0;
        cusparseCreateMatDescr(&descr);
        cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
        cusparseSetMatFillMode(descr, CUSPARSE_FILL_MODE_LOWER);
        cusparseSetMatDiagType(descr, CUSPARSE_DIAG_TYPE_UNIT);

        const cusparseOperation_t trans = CUSPARSE_OPERATION_NON_TRANSPOSE;

        const float alpha = 1.0;
        const float beta = 0.0;

#pragma omp target data use_device_ptr(xd, yd, vald, rowd, cold)
        {
          internal::check_CUDA(cusparseScsrmv(sp_handle, trans, m, n, nnz, &alpha,
                descr, vald, rowd, cold, xd, &beta,
                yd));
        }
#else
        throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
      } else {
        // MKL
#if MONOLISH_USE_MKL
        const float alpha = 1.0;
        const float beta = 0.0;
        mkl_scsrmv("N", &m, &n, &alpha, "G__C", vald, cold, rowd, rowd + 1, xd,
            &beta, yd);

        // OSS
#else
#pragma omp parallel for
        for (int i = 0; i < (int)A.get_row(); i++) {
          float ytmp = 0.0;
          for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; j++) {
            ytmp += vald[j] * xd[A.col_ind[j]];
          }
          y[i] = ytmp;
        }
#endif
      }

      logger.func_out();
    }
}

namespace blas{
  void matvec(const matrix::CRS<double> &A, const vector<double> &x, vector<double> &y) {
    Dmatvec_core(A,x,y);
  }

  void matvec(const matrix::CRS<float> &A, const vector<float> &x, vector<float> &y) {
    Smatvec_core(A,x,y);
  }
}

} // namespace monolish
