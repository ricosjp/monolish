#include "../../../../include/monolish_blas.hpp"
#include "../../../internal/monolish_internal.hpp"

namespace monolish {

// double ///////////////////
void blas::matmul(const double &a, const matrix::CRS<double> &A,
                  const matrix::Dense<double> &B, const double &b,
                  matrix::Dense<double> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(A.get_col() == B.get_row());
  assert(A.get_row() == C.get_row());
  assert(B.get_col() == C.get_col());
  assert(util::is_same_device_mem_stat(A, B, C));

  const double *vald = A.vad;
  const auto rowd = A.row_ptr.data();
  const auto cold = A.col_ind.data();

  const double *Bd = B.vad;
  double *Cd = C.vad;

  // MN = MK * KN
  const int M = (int)A.get_row();
  const int N = (int)B.get_col();
  const int K = (int)A.get_col();

  if (A.get_device_mem_stat() == true) {
#if MONOLISH_USE_NVIDIA_GPU // CUDA11 will support SpMM
    auto nnz = A.get_nnz();
    auto alpha = a;
    auto beta = b;

#pragma omp target data use_device_ptr(Bd, Cd, vald, rowd, cold)
    {
      cusparseHandle_t sp_handle;
      cusparseCreate(&sp_handle);
      cudaDeviceSynchronize();
      const cusparseOperation_t trans = CUSPARSE_OPERATION_NON_TRANSPOSE;

      cusparseSpMatDescr_t matA;
      cusparseDnMatDescr_t matB, matC;
      void *dBuffer = NULL;
      size_t buffersize = 0;

      cusparseCreateCsr(&matA, M, K, nnz, (void *)rowd, (void *)cold,
                        (void *)vald, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
      cusparseCreateDnMat(&matB, K, N, N, (void *)Bd, CUDA_R_64F,
                          CUSPARSE_ORDER_ROW);
      cusparseCreateDnMat(&matC, M, N, N, (void *)Cd, CUDA_R_64F,
                          CUSPARSE_ORDER_ROW);

      cusparseSpMM_bufferSize(sp_handle, trans, trans, &alpha, matA, matB,
                              &beta, matC, CUDA_R_64F,
                              CUSPARSE_SPMM_ALG_DEFAULT, &buffersize);

      cudaMalloc(&dBuffer, buffersize);

      cusparseSpMM(sp_handle, trans, trans, &alpha, matA, matB, &beta, matC,
                   CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer);

      cusparseDestroySpMat(matA);
      cusparseDestroyDnMat(matB);
      cusparseDestroyDnMat(matC);
      cudaFree(dBuffer);
    }
#else
    throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
  } else {
// MKL
#if MONOLISH_USE_MKL
    const double alpha = a;
    const double beta = b;

    //  sparse_matrix_t mklA;
    //  struct matrix_descr descrA;
    //  descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
    //  mkl_sparse_d_create_csr(&mklA, SPARSE_INDEX_BASE_ZERO, M, K, (int*)rowd,
    //  (int*)rowd+1, (int*)cold, (double*)vald); mkl_sparse_set_mm_hint (mklA,
    //  SPARSE_OPERATION_NON_TRANSPOSE, descrA, 100); // We haven't seen any
    //  performance improvement by using hint.
    //  mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, alpha, mklA, descrA,
    //  SPARSE_LAYOUT_ROW_MAJOR,
    //              Bd, N, K, beta, Cd, N);

    mkl_dcsrmm("N", &M, &N, &K, &alpha, "G__C", vald, cold, rowd, rowd + 1, Bd,
               &N, &beta, Cd, &N);

// OSS
#else
#if USE_AVX // avx_cpu
    const auto vecL = 4;

#pragma omp parallel for
    for (auto i = decltype(M){0}; i < M * N; i++) {
      Cd[i] = b * Cd[i];
    }

#pragma omp parallel for
    for (auto i = decltype(M){0}; i < M; i++) {
      auto start = (int)rowd[i];
      auto end = (int)rowd[i + 1];
      auto Cr = i * N;
      for (auto k = start; k < end; k++) {
        auto Br = N * cold[k];
        auto avald = a * vald[k];
        const Dreg Av = SIMD_FUNC(broadcast_sd)(&avald);
        Dreg tv, Bv, Cv;
        int j;
        for (j = 0; j < N - (vecL - 1); j += vecL) {
          auto BB = Br + j;
          auto CC = Cr + j;

          Bv = SIMD_FUNC(loadu_pd)((double *)&Bd[BB]);
          Cv = SIMD_FUNC(loadu_pd)((double *)&Cd[CC]);
          tv = SIMD_FUNC(mul_pd)(Av, Bv);
          Cv = SIMD_FUNC(add_pd)(Cv, tv);
          SIMD_FUNC(storeu_pd)((double *)&Cd[CC], Cv);
        }

        for (; j < N; j++) {
          Cd[Cr + j] += a * vald[k] * Bd[Br + j];
        }
      }
    }
#else // Scalar_cpu
#pragma omp parallel for
    for (auto j = decltype(N){0}; j < N; j++) {
      for (auto i = decltype(M){0}; i < M; i++) {
        double tmp = 0;
        auto start = (int)rowd[i];
        auto end = (int)rowd[i + 1];
        for (auto k = start; k < end; k++) {
          tmp += vald[k] * Bd[N * cold[k] + j];
        }
        Cd[i * N + j] = a * tmp + b * Cd[i * N + j];
      }
    }
#endif
#endif
  }
  logger.func_out();
}

void blas::matmul(const matrix::CRS<double> &A, const matrix::Dense<double> &B,
                  matrix::Dense<double> &C) {
  return matmul(1.0, A, B, 0.0, C);
}

// float ///////////////////
void blas::matmul(const float &a, const matrix::CRS<float> &A,
                  const matrix::Dense<float> &B, const float &b,
                  matrix::Dense<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(A.get_col() == B.get_row());
  assert(A.get_row() == C.get_row());
  assert(B.get_col() == C.get_col());
  assert(util::is_same_device_mem_stat(A, B, C));

  const float *vald = A.vad;
  const auto *rowd = A.row_ptr.data();
  const auto *cold = A.col_ind.data();

  const float *Bd = B.vad;
  float *Cd = C.vad;

  // MN = MK * KN
  const int M = A.get_row();
  const int N = B.get_col();
  const int K = A.get_col();

  if (A.get_device_mem_stat() == true) {
#if MONOLISH_USE_NVIDIA_GPU
// CUDA11.4 SpMM has bug
//#if MONOLISH_USE_OLD_CUDA // cuda10.x
#pragma omp target teams distribute parallel for
    for (auto j = decltype(N){0}; j < N; j++) {
      for (auto i = decltype(M){0}; i < M; i++) {
        float tmp = 0;
        for (auto k = rowd[i]; k < rowd[i + 1]; k++) {
          tmp += vald[k] * Bd[N * cold[k] + j];
        }
        Cd[i * N + j] = a * tmp + b * Cd[i * N + j];
      }
    }
// #else
//
//     size_t nnz = A.get_nnz();
//     const double alpha = 1.0;
//     const double beta = 0.0;
//
// #pragma omp target data use_device_ptr(Bd, Cd, vald, rowd, cold)
//     {
//       cusparseHandle_t sp_handle;
//       cusparseCreate(&sp_handle);
//       cudaDeviceSynchronize();
//       const cusparseOperation_t trans = CUSPARSE_OPERATION_NON_TRANSPOSE;
//
//       cusparseSpMatDescr_t matA;
//       cusparseDnMatDescr_t matB, matC;
//       void *dBuffer = NULL;
//       size_t buffersize = 0;
//
//       cusparseCreateCsr(&matA, M, K, nnz, (void *)rowd, (void *)cold,
//                         (void *)vald, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
//                         CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
//       cusparseCreateDnMat(&matB, K, N, N, (void *)Bd, CUDA_R_32F,
//                           CUSPARSE_ORDER_ROW);
//       cusparseCreateDnMat(&matC, M, N, N, (void *)Cd, CUDA_R_32F,
//                           CUSPARSE_ORDER_ROW);
//
//       cusparseSpMM_bufferSize(sp_handle, trans, trans, &alpha, matA, matB,
//                               &beta, matC, CUDA_R_32F,
//                               CUSPARSE_SPMM_ALG_DEFAULT, &buffersize);
//
//       cudaMalloc(&dBuffer, buffersize);
//
//       cusparseSpMM(sp_handle, trans, trans, &alpha, matA, matB, &beta, matC,
//                    CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer);
//
//       cusparseDestroySpMat(matA);
//       cusparseDestroyDnMat(matB);
//       cusparseDestroyDnMat(matC);
//       cudaFree(dBuffer);
//     }
// #endif
#else
    throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
  } else {
// MKL
#if MONOLISH_USE_MKL
    const float alpha = a;
    const float beta = b;
    mkl_scsrmm("N", &M, &N, &K, &alpha, "G__C", vald, cold, rowd, rowd + 1, Bd,
               &N, &beta, Cd, &N);

// OSS
#else
#if MONOLISH_USE_AVX // avx_cpu
                     // const int vecL = 8;

#pragma omp parallel for
    for (auto i = decltype(M){0}; i < M * N; i++) {
      Cd[i] = b * Cd[i];
    }

#pragma omp parallel for
    for (auto i = decltype(M){0}; i < M; i++) {
      auto start = (int)rowd[i];
      auto end = (int)rowd[i + 1];
      auto Cr = i * N;
      for (int k = start; k < end; k++) {
        const int Br = N * cold[k];
        auto avald = a * vald[k];
        const Sreg Av = SIMD_FUNC(broadcast_ss)(&avald);
        Sreg tv, Bv, Cv;
        int j;
        for (j = 0; j < (int)N - 31; j += 32) {
          const int BB = Br + j;
          const int CC = Cr + j;

          Bv = SIMD_FUNC(loadu_ps)((float *)&Bd[BB]);
          Cv = SIMD_FUNC(loadu_ps)((float *)&Cd[CC]);
          tv = SIMD_FUNC(mul_ps)(Av, Bv);
          Cv = SIMD_FUNC(add_ps)(Cv, tv);
          SIMD_FUNC(storeu_ps)((float *)&Cd[CC], Cv);

          Bv = SIMD_FUNC(loadu_ps)((float *)&Bd[BB + 8]);
          Cv = SIMD_FUNC(loadu_ps)((float *)&Cd[CC + 8]);
          tv = SIMD_FUNC(mul_ps)(Av, Bv);
          Cv = SIMD_FUNC(add_ps)(Cv, tv);
          SIMD_FUNC(storeu_ps)((float *)&Cd[CC + 8], Cv);

          Bv = SIMD_FUNC(loadu_ps)((float *)&Bd[BB + 16]);
          Cv = SIMD_FUNC(loadu_ps)((float *)&Cd[CC + 16]);
          tv = SIMD_FUNC(mul_ps)(Av, Bv);
          Cv = SIMD_FUNC(add_ps)(Cv, tv);
          SIMD_FUNC(storeu_ps)((float *)&Cd[Cr + j + 16], Cv);

          Bv = SIMD_FUNC(loadu_ps)((float *)&Bd[BB + 24]);
          Cv = SIMD_FUNC(loadu_ps)((float *)&Cd[CC + 24]);
          tv = SIMD_FUNC(mul_ps)(Av, Bv);
          Cv = SIMD_FUNC(add_ps)(Cv, tv);
          SIMD_FUNC(storeu_ps)((float *)&Cd[CC + 24], Cv);
        }
        for (; j < (int)N - 7; j += 8) {
          const int BB = Br + j;
          const int CC = Cr + j;

          Bv = SIMD_FUNC(loadu_ps)((float *)&Bd[BB]);
          Cv = SIMD_FUNC(loadu_ps)((float *)&Cd[CC]);
          tv = SIMD_FUNC(mul_ps)(Av, Bv);
          Cv = SIMD_FUNC(add_ps)(Cv, tv);
          SIMD_FUNC(storeu_ps)((float *)&Cd[CC], Cv);
        }
        for (; j < (int)N; j++) {
          Cd[Cr + j] += a * vald[k] * Bd[Br + j];
        }
      }
    }
#else // Scalar_cpu
#pragma omp parallel for
    for (auto j = decltype(N){0}; j < N; j++) {
      for (auto i = decltype(M){0}; i < M; i++) {
        float tmp = 0;
        auto start = (int)rowd[i];
        auto end = (int)rowd[i + 1];
        for (auto k = start; k < end; k++) {
          tmp += vald[k] * Bd[N * cold[k] + j];
        }
        Cd[i * N + j] = a * tmp + b * Cd[i * N + j];
      }
    }
#endif
#endif
  }
  logger.func_out();
}

void blas::matmul(const matrix::CRS<float> &A, const matrix::Dense<float> &B,
                  matrix::Dense<float> &C) {
  return matmul(1.0, A, B, 0.0, C);
}

} // namespace monolish
