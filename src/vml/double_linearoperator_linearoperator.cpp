#include "../../include/monolish_vml.hpp"
#include "../../include/monolish_blas.hpp"
#include "../internal/monolish_internal.hpp"

namespace monolish {

///////////////////////////////////////////
// LinearOperator /////////////////////////
///////////////////////////////////////////

void vml::add(const matrix::LinearOperator<double>& A, const matrix::LinearOperator<double>& B,
             matrix::LinearOperator<double>& C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_row() != B.get_row() && A.get_row() != C.get_row()) {
    throw std::runtime_error("error A.row != B.row != C.row");
  }
  if (A.get_col() != B.get_col() && A.get_col() != C.get_col()) {
    throw std::runtime_error("error A.col != B.col != C.col");
  }
  if (A.get_device_mem_stat() != B.get_device_mem_stat() ||
      A.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error matrix get_device_mem_stat() is not same");
  }

  if(A.get_matvec_init_flag() != B.get_matvec_init_flag()){
    throw std::runtime_error("error A.matvec_init_flag != B.matvec_init_flag");
  }

  if(A.get_rmatvec_init_flag() != B.get_rmatvec_init_flag()){
    throw std::runtime_error("error A.rmatvec_init_flag != B.rmatvec_init_flag");
  }

  if(A.get_matvec_init_flag()){
    C.set_matvec(
      [&](const vector<double>& VEC){
        vector<double> vec(A.get_row(), 0.0), vec_tmp(A.get_row(), 0.0);
        if(A.get_device_mem_stat()){
          util::send(vec, vec_tmp);
        }
        blas::matvec(A, VEC, vec);
        blas::matvec(B, VEC, vec_tmp);
        blas::axpy(1.0, vec_tmp, vec);
        if(A.get_device_mem_stat()){
          util::device_free(vec_tmp);
        }
        return vec;
      }
    );
  }

  if(A.get_rmatvec_init_flag()){
    C.set_rmatvec(
      [&](const vector<double>& VEC){
        vector<double> vec(A.get_col(), 0.0), vec_tmp(A.get_col(), 0.0);
        if(A.get_device_mem_stat()){
          util::send(vec, vec_tmp);
        }
        blas::rmatvec(A, VEC, vec);
        blas::rmatvec(B, VEC, vec_tmp);
        blas::axpy(1.0, vec_tmp, vec);
        if(A.get_device_mem_stat()){
          util::device_free(vec_tmp);
        }
        return vec;
      }
    );
  }

  logger.func_out();
}

void vml::sub(const matrix::LinearOperator<double>& A, const matrix::LinearOperator<double>& B,
             matrix::LinearOperator<double>& C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_row() != B.get_row() && A.get_row() != C.get_row()) {
    throw std::runtime_error("error A.row != B.row != C.row");
  }
  if (A.get_col() != B.get_col() && A.get_col() != C.get_col()) {
    throw std::runtime_error("error A.col != B.col != C.col");
  }
  if (A.get_device_mem_stat() != B.get_device_mem_stat() ||
      A.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error matrix get_device_mem_stat() is not same");
  }

  if(A.get_matvec_init_flag() != B.get_matvec_init_flag()){
    throw std::runtime_error("error A.matvec_init_flag != B.matvec_init_flag");
  }

  if(A.get_rmatvec_init_flag() != B.get_rmatvec_init_flag()){
    throw std::runtime_error("error A.rmatvec_init_flag != B.rmatvec_init_flag");
  }

  if(A.get_matvec_init_flag()){
    C.set_matvec(
      [&](const vector<double>& VEC){
        vector<double> vec(A.get_row(), 0.0), vec_tmp(A.get_row(), 0.0);
        if(A.get_device_mem_stat()){
          util::send(vec, vec_tmp);
        }
        blas::matvec(A, VEC, vec);
        blas::matvec(B, VEC, vec_tmp);
        blas::axpy(-1.0, vec_tmp, vec);
        if(A.get_device_mem_stat()){
          util::device_free(vec_tmp);
        }
        return vec;
      }
    );
  }

  if(A.get_rmatvec_init_flag()){
    C.set_rmatvec(
      [&](const vector<double>& VEC){
        vector<double> vec(A.get_col(), 0.0), vec_tmp(A.get_col(), 0.0);
        if(A.get_device_mem_stat()){
          util::send(vec, vec_tmp);
        }
        blas::rmatvec(A, VEC, vec);
        blas::rmatvec(B, VEC, vec_tmp);
        blas::axpy(-1.0, vec_tmp, vec);
        if(A.get_device_mem_stat()){
          util::device_free(vec_tmp);
        }
        return vec;
      }
    );
  }

  logger.func_out();
}

} // namespace monolish

