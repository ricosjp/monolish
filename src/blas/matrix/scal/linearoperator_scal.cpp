#include "../../../../include/monolish_blas.hpp"
#include "../../../internal/monolish_internal.hpp"

namespace monolish {

void blas::mscal(const double alpha, matrix::LinearOperator<double> &A){
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  if(A.get_matvec_init_flag()){
    A.set_matvec(
      [&](const vector<double>& VEC){
        vector<double> vec(A.get_row(), 0.0);
        blas::matvec(A, VEC, vec);
        blas::axpy(alpha-1.0, vec, vec);
        return vec;
      }
    );
  }
  if(A.get_rmatvec_init_flag()){
    A.set_rmatvec(
      [&](const vector<double>& VEC){
        vector<double> vec(A.get_col(), 0.0);
        blas::rmatvec(A, VEC, vec);
        blas::axpy(alpha-1.0, vec, vec);
        return vec;
      }
    );
  }

  logger.func_out();
}

void blas::mscal(const float alpha, matrix::LinearOperator<float> &A){
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  if(A.get_matvec_init_flag()){
    A.set_matvec(
      [&](const vector<float>& VEC){
        vector<float> vec(A.get_row(), 0.0);
        blas::matvec(A, VEC, vec);
        blas::axpy(alpha-1.0, vec, vec);
        return vec;
      }
    );
  }
  if(A.get_rmatvec_init_flag()){
    A.set_rmatvec(
      [&](const vector<float>& VEC){
        vector<float> vec(A.get_col(), 0.0);
        blas::rmatvec(A, VEC, vec);
        blas::axpy(alpha-1.0, vec, vec);
        return vec;
      }
    );
  }

  logger.func_out();
}
} // namespace monolish
