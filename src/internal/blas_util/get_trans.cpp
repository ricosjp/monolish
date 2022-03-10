#include "../monolish_internal.hpp"

namespace monolish::internal {

    CBLAS_TRANSPOSE get_cblas_trans(bool flag){
        if(flag ==true){
            return CblasTrans;
        }
        else{
            return CblasNoTrans;
        }
    }

#ifdef MONOLISH_USE_NVIDIA_GPU
    cusparseOperation_t get_cuspasrse_trans(bool flag){
        if(flag ==true){
            return CUSPARSE_OPERATION_TRANSPOSE;
        }
        else{
            return CUSPARSE_OPERATION_NON_TRANSPOSE;
        }
    }

    cublasOperation_t get_cublas_trans(bool flag){
        if(flag ==true){
            return CUBLAS_OP_T;
        }
        else{
            return CUBLAS_OP_N;
        }
    }
#endif

}
