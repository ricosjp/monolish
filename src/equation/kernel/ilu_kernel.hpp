#pragma once

namespace monolish {

namespace {

#if MONOLISH_USE_NVIDIA_GPU
void cusolver_ilu_create_descr(
        matrix::CRS<double> &A,
        cusparseMatDescr_t &descr_M,
        csrilu02Info_t &info_M,
        cusparseMatDescr_t &descr_L,
        csrsv2Info_t &info_L,
        cusparseMatDescr_t &descr_U,
        csrsv2Info_t &info_U,
        const cusparseHandle_t &handle) {

  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  auto M = A.get_row();
  auto nnz = A.get_nnz();
  int* d_csrRowPtr = A.row_ptr.data();
  int* d_csrColInd = A.col_ind.data();
  auto* d_csrVal = A.val.data();

#pragma omp target data use_device_ptr(d_csrVal, d_csrRowPtr, d_csrColInd)
  {
      cusparseCreateMatDescr(&descr_M);
      cusparseSetMatIndexBase(descr_M, CUSPARSE_INDEX_BASE_ZERO);
      cusparseSetMatType(descr_M, CUSPARSE_MATRIX_TYPE_GENERAL);

      cusparseCreateMatDescr(&descr_L);
      cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ZERO);
      cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL);
      cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER);
      cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_UNIT);

      cusparseCreateMatDescr(&descr_U);
      cusparseSetMatIndexBase(descr_U, CUSPARSE_INDEX_BASE_ZERO);
      cusparseSetMatType(descr_U, CUSPARSE_MATRIX_TYPE_GENERAL);
      cusparseSetMatFillMode(descr_U, CUSPARSE_FILL_MODE_UPPER);
      cusparseSetMatDiagType(descr_U, CUSPARSE_DIAG_TYPE_NON_UNIT);

      // step 2: create a empty info structure
      // we need one info for csrilu02 and two info's for csrsv2
      cusparseCreateCsrilu02Info(&info_M);
      cusparseCreateCsrsv2Info(&info_L);
      cusparseCreateCsrsv2Info(&info_U);
  }

  logger.func_out();
}

int cusolver_ilu_get_buffersize(
        matrix::CRS<double> &A,
        const cusparseMatDescr_t &descr_M,
        const csrilu02Info_t &info_M,
        const cusparseMatDescr_t &descr_L,
        const csrsv2Info_t &info_L,
        const cusparseOperation_t &trans_L,
        const cusparseMatDescr_t &descr_U,
        const csrsv2Info_t &info_U,
        const cusparseOperation_t &trans_U,
        const cusparseHandle_t &handle) {

  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  auto M = A.get_row();
  auto nnz = A.get_nnz();
  int* d_csrRowPtr = A.row_ptr.data();
  int* d_csrColInd = A.col_ind.data();
  auto* d_csrVal = A.val.data();

  int bufsize;
  int bufsize_M;
  int bufsize_L;
  int bufsize_U;

#pragma omp target data use_device_ptr(d_csrVal, d_csrRowPtr, d_csrColInd)
  {
      cusparseDcsrilu02_bufferSize(handle, M, nnz,
              descr_M, d_csrVal, d_csrRowPtr, d_csrColInd, info_M, &bufsize_M);
      cusparseDcsrsv2_bufferSize(handle, trans_L, M, nnz,
              descr_L, d_csrVal, d_csrRowPtr, d_csrColInd, info_L, &bufsize_L);
      cusparseDcsrsv2_bufferSize(handle, trans_U, M, nnz,
              descr_U, d_csrVal, d_csrRowPtr, d_csrColInd, info_U, &bufsize_U);

      bufsize = std::max(bufsize_M, std::max(bufsize_L, bufsize_U));
  }

  logger.func_out();
  return bufsize;
}
#endif

} // namespace
} // namespace monolish
