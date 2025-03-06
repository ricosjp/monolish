#pragma once
#include "../../../../include/monolish_blas.hpp"
#include "../../../internal/monolish_internal.hpp"

#define CHECK_CUDA(func)                                                       \
  {                                                                            \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
      printf("CUDA API failed at line %d with error: %s (%d)\n", __LINE__,     \
             cudaGetErrorString(status), status);                              \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

#define CHECK_CUSPARSE(func)                                                   \
  {                                                                            \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
      printf("CUSPARSE API failed at line %d with error: %s (%d)\n", __LINE__, \
             cusparseGetErrorString(status), status);                          \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

namespace monolish {
namespace {

using Float = float;
auto Cutype = CUDA_R_32F;

int spmm_csr_example(void) {
  // Host problem definition
  int A_num_rows = 5;
  int A_num_cols = 4;
  int A_nnz = 9;
  int B_num_rows = A_num_cols;
  int B_num_cols = 3;
  int B_size = B_num_rows * B_num_cols;
  int C_size = A_num_rows * B_num_cols;
  int hA_csrOffsets[] = {0, 3, 4, 7, 9, 9};
  int hA_columns[] = {0, 2, 3, 1, 0, 2, 3, 1, 3};
  Float hA_values[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
  Float hB[] = {1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f,
                7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  Float hC[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  Float hC_result[] = {45.0f,  51.0f,  57.0f,  16.0f,  20.0f,
                       24.0f,  117.0f, 135.0f, 153.0f, 122.0f,
                       139.0f, 156.0f, 0.0f,   0.0f,   0.0f};
  Float alpha = 1.0f;
  Float beta = 0.0f;
  //--------------------------------------------------------------------------
  // Device memory management
  int *dA_csrOffsets, *dA_columns;
  Float *dA_values, *dB, *dC;
  CHECK_CUDA(
      cudaMalloc((void **)&dA_csrOffsets, (A_num_rows + 1) * sizeof(int)))
  CHECK_CUDA(cudaMalloc((void **)&dA_columns, A_nnz * sizeof(int)))
  CHECK_CUDA(cudaMalloc((void **)&dA_values, A_nnz * sizeof(Float)))
  CHECK_CUDA(cudaMalloc((void **)&dB, B_size * sizeof(Float)))
  CHECK_CUDA(cudaMalloc((void **)&dC, C_size * sizeof(Float)))

  CHECK_CUDA(cudaMemcpy(dA_csrOffsets, hA_csrOffsets,
                        (A_num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice))
  CHECK_CUDA(cudaMemcpy(dA_columns, hA_columns, A_nnz * sizeof(int),
                        cudaMemcpyHostToDevice))
  CHECK_CUDA(cudaMemcpy(dA_values, hA_values, A_nnz * sizeof(Float),
                        cudaMemcpyHostToDevice))
  CHECK_CUDA(cudaMemcpy(dB, hB, B_size * sizeof(Float), cudaMemcpyHostToDevice))
  CHECK_CUDA(cudaMemcpy(dC, hC, C_size * sizeof(Float), cudaMemcpyHostToDevice))
  //--------------------------------------------------------------------------
  // CUSPARSE APIs
  cusparseHandle_t handle = NULL;
  cusparseSpMatDescr_t matA;
  cusparseDnMatDescr_t matB, matC;
  void *dBuffer = NULL;
  size_t bufferSize = 0;
  const cusparseOperation_t trans = CUSPARSE_OPERATION_NON_TRANSPOSE;
  CHECK_CUSPARSE(cusparseCreate(&handle))
  // Create sparse matrix A in CSR format
  CHECK_CUSPARSE(cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                   dA_csrOffsets, dA_columns, dA_values,
                                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_BASE_ZERO, Cutype))
  // Create dense matrix B
  CHECK_CUSPARSE(cusparseCreateDnMat(&matB, A_num_cols, B_num_cols, B_num_cols,
                                     dB, Cutype, CUSPARSE_ORDER_ROW))
  // Create dense matrix C
  CHECK_CUSPARSE(cusparseCreateDnMat(&matC, A_num_rows, B_num_cols, B_num_cols,
                                     dC, Cutype, CUSPARSE_ORDER_ROW))
  // allocate an external buffer if needed
  CHECK_CUSPARSE(cusparseSpMM_bufferSize(
      handle, trans, trans, &alpha, matA, matB, &beta, matC, Cutype,
      CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize))
  CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize))

  // execute SpMM
  CHECK_CUSPARSE(cusparseSpMM(handle, trans, trans, &alpha, matA, matB, &beta,
                              matC, Cutype, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer))

  // destroy matrix/vector descriptors
  CHECK_CUSPARSE(cusparseDestroySpMat(matA))
  CHECK_CUSPARSE(cusparseDestroyDnMat(matB))
  CHECK_CUSPARSE(cusparseDestroyDnMat(matC))
  CHECK_CUSPARSE(cusparseDestroy(handle))
  //--------------------------------------------------------------------------
  // device result check
  CHECK_CUDA(cudaMemcpy(hC, dC, C_size * sizeof(Float), cudaMemcpyDeviceToHost))

  //--------------------------------------------------------------------------
  // device memory deallocation
  CHECK_CUDA(cudaFree(dBuffer))
  CHECK_CUDA(cudaFree(dA_csrOffsets))
  CHECK_CUDA(cudaFree(dA_columns))
  CHECK_CUDA(cudaFree(dA_values))
  CHECK_CUDA(cudaFree(dB))
  CHECK_CUDA(cudaFree(dC))

  int correct = 1;
  for (int i = 0; i < A_num_rows; i++) {
    for (int j = 0; j < B_num_cols; j++) {
      if (hC[i * B_num_cols + j] != hC_result[i * B_num_cols + j]) {
        correct = 0; // direct Floating point comparison is not reliable
        printf("(%d %d) : %f vs %f\n", i, j, hC[i * B_num_cols + j],
               hC_result[i * B_num_cols + j]);
      }
    }
  }
  if (correct)
    printf("spmm_csr_example test PASSED\n");
  else
    printf("spmm_csr_example test FAILED: wrong result\n");
  return EXIT_SUCCESS;
};

// double ///////////////////
template <typename MAT1, typename MAT2>
void CRS_Dense_Dmatmul_core(const double &a, const matrix::CRS<double> &A,
                            const MAT1 &B, const double &b, MAT2 &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // spmm_csr_example();
  // exit(EXIT_FAILURE);

  // err
  assert(A.get_col() == B.get_row());
  assert(A.get_row() == C.get_row());
  assert(B.get_col() == C.get_col());
  assert(util::is_same_device_mem_stat(A, B, C));

  const double *vald = A.begin();
  const int *rowd = A.row_ptr.data();
  const int *cold = A.col_ind.data();

  const double *Bd = B.begin();
  double *Cd = C.begin();

  // MN = MK * KN
  const int M = (int)A.get_row();
  const int N = (int)B.get_col();
  const int K = (int)A.get_col();

  if (A.get_device_mem_stat() == true) {
#if MONOLISH_USE_NVIDIA_GPU // CUDA11 will support SpMM
    int nnz = A.get_nnz();
    double alpha = a;
    double beta = b;

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

    /*
    //--------------------------------------------------------------------------
    // Device memory management
    int   *drowd, *dcold;
    double *dvald, *dBd, *dCd;
    CHECK_CUDA( cudaMalloc((void**) &drowd,
                           (M + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dcold,       nnz * sizeof(int))    )
    CHECK_CUDA( cudaMalloc((void**) &dvald,       nnz * sizeof(double))  )
    CHECK_CUDA( cudaMalloc((void**) &dBd,         K*N * sizeof(double)) )
    CHECK_CUDA( cudaMalloc((void**) &dCd,         M*N * sizeof(double)) )

    CHECK_CUDA( cudaMemcpy(drowd, rowd,
                           (M + 1) * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dcold, cold, nnz * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dvald, vald, nnz * sizeof(double),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dBd, Bd, K*N * sizeof(double),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dCd, Cd, M*N * sizeof(double),
                           cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    cudaDeviceSynchronize();
    const cusparseOperation_t trans = CUSPARSE_OPERATION_NON_TRANSPOSE;

    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;

    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, M, K, nnz,
                                      drowd, dcold, dvald,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F) )
    // Create dense matrix B
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB, K, N, N, dBd,
                                        CUDA_R_64F, CUSPARSE_ORDER_ROW) )
    // Create dense matrix C
    CHECK_CUSPARSE( cusparseCreateDnMat(&matC, M, N, N, dCd,
                                        CUDA_R_64F, CUSPARSE_ORDER_ROW) )
    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
                                 handle,
                                 trans, trans,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_64F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    // execute SpMM
    CHECK_CUSPARSE( cusparseSpMM(handle,
                                 trans, trans,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_64F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, dBuffer) )

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matC) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    //--------------------------------------------------------------------------
    // device result check
    CHECK_CUDA( cudaMemcpy(Cd, dCd, M*N * sizeof(double),
                           cudaMemcpyDeviceToHost) )
    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA( cudaFree(dBuffer) )
    CHECK_CUDA( cudaFree(drowd) )
    CHECK_CUDA( cudaFree(dcold) )
    CHECK_CUDA( cudaFree(dvald) )
    CHECK_CUDA( cudaFree(dBd) )
    CHECK_CUDA( cudaFree(dCd) )
    */

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

// float ///////////////////
template <typename MAT1, typename MAT2>
void CRS_Dense_Smatmul_core(const float &a, const matrix::CRS<float> &A,
                            const MAT1 &B, const float &b, MAT2 &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(A.get_col() == B.get_row());
  assert(A.get_row() == C.get_row());
  assert(B.get_col() == C.get_col());
  assert(util::is_same_device_mem_stat(A, B, C));

  const float *vald = A.begin();
  const int *rowd = A.row_ptr.data();
  const int *cold = A.col_ind.data();

  const float *Bd = B.begin();
  float *Cd = C.begin();

  // MN = MK * KN
  const int M = A.get_row();
  const int N = B.get_col();
  const int K = A.get_col();

  if (A.get_device_mem_stat() == true) {
#if MONOLISH_USE_NVIDIA_GPU
// CUDA11.4 SpMM has bug ?
#if MONOLISH_USE_OLD_CUDA // cuda10.x or cuda 11.4
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
#else

    int nnz = A.get_nnz();
    const float alpha = a;
    const float beta = b;

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
                        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
      cusparseCreateDnMat(&matB, K, N, N, (void *)Bd, CUDA_R_32F,
                          CUSPARSE_ORDER_ROW);
      cusparseCreateDnMat(&matC, M, N, N, (void *)Cd, CUDA_R_32F,
                          CUSPARSE_ORDER_ROW);

      cusparseSpMM_bufferSize(sp_handle, trans, trans, &alpha, matA, matB,
                              &beta, matC, CUDA_R_32F,
                              CUSPARSE_SPMM_ALG_DEFAULT, &buffersize);

      cudaMalloc(&dBuffer, buffersize);

      cusparseSpMM(sp_handle, trans, trans, &alpha, matA, matB, &beta, matC,
                   CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer);

      cusparseDestroySpMat(matA);
      cusparseDestroyDnMat(matB);
      cusparseDestroyDnMat(matC);
      cudaFree(dBuffer);
    }
#endif

    /*
    //--------------------------------------------------------------------------
    // Device memory management
    int   *dA_csrOffsets, *dA_columns;
    float *dA_values, *dB, *dC;
    CHECK_CUDA( cudaMalloc((void**) &dA_csrOffsets,
                           (M + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dA_columns, nnz * sizeof(int))    )
    CHECK_CUDA( cudaMalloc((void**) &dA_values,  nnz * sizeof(float))  )
    CHECK_CUDA( cudaMalloc((void**) &dB,         K*N * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &dC,         M*N * sizeof(float)) )

    CHECK_CUDA( cudaMemcpy(dA_csrOffsets, rowd,
                           (M + 1) * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_columns, cold, nnz * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_values, vald, nnz * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB, Bd, K*N * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC, Cd, M*N * sizeof(float),
                           cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    const cusparseOperation_t trans = CUSPARSE_OPERATION_NON_TRANSPOSE;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, M, K, nnz,
                                      dA_csrOffsets, dA_columns, dA_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    // Create dense matrix B
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB, K, N, N, dB,
                                        CUDA_R_32F, CUSPARSE_ORDER_ROW) )
    // Create dense matrix C
    CHECK_CUSPARSE( cusparseCreateDnMat(&matC, M, N, N, dC,
                                        CUDA_R_32F, CUSPARSE_ORDER_ROW) )
    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
                                 handle,
                                 trans, trans,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    // execute SpMM
    CHECK_CUSPARSE( cusparseSpMM(handle,
                                 trans, trans,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, dBuffer) )

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matC) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    //--------------------------------------------------------------------------
    // device result check
    CHECK_CUDA( cudaMemcpy(Cd, dC, M*N * sizeof(float),
                           cudaMemcpyDeviceToHost) )
    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA( cudaFree(dBuffer) )
    CHECK_CUDA( cudaFree(dA_csrOffsets) )
    CHECK_CUDA( cudaFree(dA_columns) )
    CHECK_CUDA( cudaFree(dA_values) )
    CHECK_CUDA( cudaFree(dB) )
    CHECK_CUDA( cudaFree(dC) )
    */
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

} // namespace
} // namespace monolish
