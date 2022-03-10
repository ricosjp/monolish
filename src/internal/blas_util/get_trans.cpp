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
    }

    cublasOperation_t get_cublas_trans(bool flag){
    }
#endif

}
