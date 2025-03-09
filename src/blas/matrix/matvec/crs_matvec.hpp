#pragma once

namespace monolish {

namespace {
// double ///////////////////
template <typename VEC1, typename VEC2>
void Dmatvec_core(const double alpha, const matrix::CRS<double> &A,
                  const VEC1 &x, const double beta, VEC2 &y, bool transA) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err, M = MN * N
  if (transA) {
    assert(A.get_row() == x.size());
    assert(A.get_col() == y.size());
  } else {
    assert(A.get_row() == y.size());
    assert(A.get_col() == x.size());
  }
  assert(util::is_same_device_mem_stat(A, x, y));

  const double *vald = A.begin();
  const double *xd = x.begin();
  const auto *rowd = A.row_ptr.data();
  const auto *cold = A.col_ind.data();
  double *yd = y.begin();

  if (A.get_device_mem_stat() == true) {
#if MONOLISH_USE_NVIDIA_GPU // gpu
    auto m = A.get_row();
    auto n = A.get_col();
    auto xn = x.size();
    auto yn = y.size();
    const auto nnz = A.get_nnz();

#pragma omp target data use_device_ptr(xd, yd, vald, rowd, cold)
    {
      cusparseSpMatDescr_t matA;
      cusparseDnVecDescr_t vecX, vecY;

      cusparseHandle_t sp_handle;
      cusparseCreate(&sp_handle);
      cudaDeviceSynchronize();

      cusparseCreateCsr(&matA, m, n, nnz, (void *)rowd, (void *)cold,
                        (void *)vald, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
      cusparseCreateDnVec(&vecX, xn, (void *)(xd), CUDA_R_64F);
      cusparseCreateDnVec(&vecY, yn, (void *)(yd), CUDA_R_64F);

      void *buffer = NULL;
      size_t buffersize = 0;
      cusparseSpMV_bufferSize(sp_handle, internal::get_cuspasrse_trans(transA),
                              &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                              CUSPARSE_MV_ALG_DEFAULT, &buffersize);
      cudaMalloc(&buffer, buffersize);
      cusparseSpMV(sp_handle, internal::get_cuspasrse_trans(transA), &alpha,
                   matA, vecX, &beta, vecY, CUDA_R_64F, CUSPARSE_MV_ALG_DEFAULT,
                   buffer);
      cusparseDestroySpMat(matA);
      cusparseDestroyDnVec(vecX);
      cusparseDestroyDnVec(vecY);
      cudaFree(buffer);
    }
#else
    throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
  } else {
    // MKL
#if MONOLISH_USE_MKL
    auto m = A.get_row();
    auto n = A.get_col();

    sparse_matrix_t mklA;
    struct matrix_descr descrA;
    descrA.type = SPARSE_MATRIX_TYPE_GENERAL;

    mkl_sparse_d_create_csr(&mklA, SPARSE_INDEX_BASE_ZERO, m, n, (int *)rowd,
                            (int *)rowd + 1, (int *)cold, (double *)vald);
    // mkl_sparse_set_mv_hint (mklA, SPARSE_OPERATION_NON_TRANSPOSE, descrA,
    // 100); // We haven't seen any performance improvement by using hint.
    mkl_sparse_d_mv(internal::get_sparseblas_trans(transA), alpha, mklA, descrA,
                    xd, beta, yd);

    // OSS
#else
    if (transA == true) {
#pragma omp parallel for
      for (auto i = decltype(y.size()){0}; i < y.size(); i++) {
        yd[i] = 0.0;
      }

      for (auto i = decltype(A.get_row()){0}; i < A.get_row(); i++) {
        for (auto j = rowd[i]; j < rowd[i + 1]; j++) {
          yd[cold[j]] += vald[j] * xd[i];
        }
      }

    } else {
#pragma omp parallel for
      for (auto i = decltype(A.get_row()){0}; i < A.get_row(); i++) {
        double ytmp = 0.0;
        for (auto j = rowd[i]; j < rowd[i + 1]; j++) {
          ytmp += vald[j] * xd[cold[j]];
        }
        yd[i] = ytmp;
      }
    }
#endif
  }

  logger.func_out();
}

// float ///////////////////
template <typename VEC1, typename VEC2>
void Smatvec_core(const float alpha, const matrix::CRS<float> &A, const VEC1 &x,
                  const float beta, VEC2 &y, bool transA) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err, M = MN * N
  if (transA) {
    assert(A.get_row() == x.size());
    assert(A.get_col() == y.size());
  } else {
    assert(A.get_row() == y.size());
    assert(A.get_col() == x.size());
  }
  assert(util::is_same_device_mem_stat(A, x, y));

  const float *vald = A.begin();
  const float *xd = x.begin();
  const auto *rowd = A.row_ptr.data();
  const auto *cold = A.col_ind.data();
  float *yd = y.begin();

  if (A.get_device_mem_stat() == true) {
#if MONOLISH_USE_NVIDIA_GPU // gpu
    const auto m = A.get_row();
    const auto n = A.get_col();
    auto xn = x.size();
    auto yn = y.size();
    const auto nnz = A.get_nnz();

#pragma omp target data use_device_ptr(xd, yd, vald, rowd, cold)
    {
      cusparseSpMatDescr_t matA;
      cusparseDnVecDescr_t vecX, vecY;

      cusparseHandle_t sp_handle;
      cusparseCreate(&sp_handle);
      cudaDeviceSynchronize();

      cusparseCreateCsr(&matA, m, n, nnz, (void *)rowd, (void *)cold,
                        (void *)vald, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
      cusparseCreateDnVec(&vecX, xn, (void *)(xd), CUDA_R_32F);
      cusparseCreateDnVec(&vecY, yn, (void *)(yd), CUDA_R_32F);

      void *buffer = NULL;
      size_t buffersize = 0;
      cusparseSpMV_bufferSize(sp_handle, internal::get_cuspasrse_trans(transA),
                              &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                              CUSPARSE_MV_ALG_DEFAULT, &buffersize);
      cudaMalloc(&buffer, buffersize);
      cusparseSpMV(sp_handle, internal::get_cuspasrse_trans(transA), &alpha,
                   matA, vecX, &beta, vecY, CUDA_R_32F, CUSPARSE_MV_ALG_DEFAULT,
                   buffer);
      cusparseDestroySpMat(matA);
      cusparseDestroyDnVec(vecX);
      cusparseDestroyDnVec(vecY);
      cudaFree(buffer);
    }
#else
    throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
  } else {
    // MKL
#if MONOLISH_USE_MKL
    const auto m = A.get_row();
    const auto n = A.get_col();

    sparse_matrix_t mklA;
    struct matrix_descr descrA;
    descrA.type = SPARSE_MATRIX_TYPE_GENERAL;

    mkl_sparse_s_create_csr(&mklA, SPARSE_INDEX_BASE_ZERO, m, n, (int *)rowd,
                            (int *)rowd + 1, (int *)cold, (float *)vald);
    // mkl_sparse_set_mv_hint (mklA, SPARSE_OPERATION_NON_TRANSPOSE, descrA,
    // 100); // We haven't seen any performance improvement by using hint.
    mkl_sparse_s_mv(internal::get_sparseblas_trans(transA), alpha, mklA, descrA,
                    xd, beta, yd);

    // OSS
#else
    if (transA == true) {
#pragma omp parallel for
      for (auto i = decltype(y.size()){0}; i < y.size(); i++) {
        yd[i] = 0.0;
      }

      for (auto i = decltype(A.get_row()){0}; i < A.get_row(); i++) {
        for (auto j = rowd[i]; j < rowd[i + 1]; j++) {
          yd[cold[j]] += vald[j] * xd[i];
        }
      }
    } else {
#pragma omp parallel for
      for (auto i = decltype(A.get_row()){0}; i < A.get_row(); i++) {
        float ytmp = 0.0;
        for (auto j = rowd[i]; j < rowd[i + 1]; j++) {
          ytmp += vald[j] * xd[cold[j]];
        }
        yd[i] = ytmp;
      }
    }
#endif
  }

  logger.func_out();
}
} // namespace
} // namespace monolish
