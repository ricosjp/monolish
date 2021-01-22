#include "../../include/monolish_vml.hpp"
#include "../../include/monolish_blas.hpp"
#include "../internal/monolish_internal.hpp"

namespace monolish {

/////////////////////////////////////////////
// LinearOperator ///////////////////////////
/////////////////////////////////////////////

void vml::add(const matrix::LinearOperator<float> &A, const float &alpha,
              matrix::LinearOperator<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_row() != C.get_row()) {
    throw std::runtime_error("error A.row != B.row != C.row");
  }
  if (A.get_col() != C.get_col()) {
    throw std::runtime_error("error A.col != B.col != C.col");
  }
  if (A.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error vector get_device_mem_stat() is not same");
  }

  if(A.get_matvec_init_flag()){
    C.set_matvec(
      [&](const vector<float>& VEC){
        vector<float> vec(A.get_row(), 0.0), vec_tmp(A.get_row(), 0.0);
        if(A.get_device_mem_stat()){
          util::send(vec, vec_tmp);
        }
        blas::matvec(A, VEC, vec_tmp);
        vml::add(vec_tmp, alpha*blas::sum(VEC), vec);
        if(A.get_device_mem_stat()){
          util::device_free(vec_tmp);
        }
        return vec;
      }
    );
  }
  if(A.get_rmatvec_init_flag()){
    C.set_rmatvec(
      [&](const vector<float>& VEC){
        vector<float> vec(A.get_col(), 0.0), vec_tmp(A.get_col(), 0.0);
        if(A.get_device_mem_stat()){
          util::send(vec, vec_tmp);
        }
        blas::rmatvec(A, VEC, vec_tmp);
        vml::add(vec_tmp, alpha*blas::sum(VEC), vec);
        if(A.get_device_mem_stat()){
          util::device_free(vec_tmp);
        }
        return vec;
      }
    );
  }

  logger.func_out();
}

void vml::sub(const matrix::LinearOperator<float> &A, const float &alpha,
              matrix::LinearOperator<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_row() != C.get_row()) {
    throw std::runtime_error("error A.row != B.row != C.row");
  }
  if (A.get_col() != C.get_col()) {
    throw std::runtime_error("error A.col != B.col != C.col");
  }
  if (A.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error vector get_device_mem_stat() is not same");
  }

  if(A.get_matvec_init_flag()){
    C.set_matvec(
      [&](const vector<float>& VEC){
        vector<float> vec(A.get_row(), 0.0), vec_tmp(A.get_row(), 0.0);
        if(A.get_device_mem_stat()){
          util::send(vec, vec_tmp);
        }
        blas::matvec(A, VEC, vec_tmp);
        vml::sub(vec_tmp, alpha*blas::sum(VEC), vec);
        if(A.get_device_mem_stat()){
          util::device_free(vec_tmp);
        }
        return vec;
      }
    );
  }
  if(A.get_rmatvec_init_flag()){
    C.set_rmatvec(
      [&](const vector<float>& VEC){
        vector<float> vec(A.get_col(), 0.0), vec_tmp(A.get_col(), 0.0);
        if(A.get_device_mem_stat()){
          util::send(vec, vec_tmp);
        }
        blas::rmatvec(A, VEC, vec_tmp);
        vml::sub(vec_tmp, alpha*blas::sum(VEC), vec);
        if(A.get_device_mem_stat()){
          util::device_free(vec_tmp);
        }
        return vec;
      }
    );
  }

  logger.func_out();
}

void vml::mul(const matrix::LinearOperator<float> &A, const float &alpha,
              matrix::LinearOperator<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_row() != C.get_row()) {
    throw std::runtime_error("error A.row != B.row != C.row");
  }
  if (A.get_col() != C.get_col()) {
    throw std::runtime_error("error A.col != B.col != C.col");
  }
  if (A.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error vector get_device_mem_stat() is not same");
  }

  if(A.get_matvec_init_flag()){
    C.set_matvec(
      [&](const vector<float>& VEC){
        vector<float> vec(A.get_row(), 0.0), vec_tmp(A.get_row(), 0.0);
        if(A.get_device_mem_stat()){
          util::send(vec, vec_tmp);
        }
        blas::matvec(A, VEC, vec_tmp);
        vml::mul(vec_tmp, alpha, vec);
        if(A.get_device_mem_stat()){
          util::device_free(vec_tmp);
        }
        return vec;
      }
    );
  }
  if(A.get_rmatvec_init_flag()){
    C.set_rmatvec(
      [&](const vector<float>& VEC){
        vector<float> vec(A.get_col(), 0.0), vec_tmp(A.get_col(), 0.0);
        if(A.get_device_mem_stat()){
          util::send(vec, vec_tmp);
        }
        blas::rmatvec(A, VEC, vec_tmp);
        vml::mul(vec_tmp, alpha, vec);
        if(A.get_device_mem_stat()){
          util::device_free(vec_tmp);
        }
        return vec;
      }
    );
  }

  logger.func_out();
}

void vml::div(const matrix::LinearOperator<float> &A, const float &alpha,
              matrix::LinearOperator<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_row() != C.get_row()) {
    throw std::runtime_error("error A.row != B.row != C.row");
  }
  if (A.get_col() != C.get_col()) {
    throw std::runtime_error("error A.col != B.col != C.col");
  }
  if (A.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error vector get_device_mem_stat() is not same");
  }

  if(A.get_matvec_init_flag()){
    C.set_matvec(
      [&](const vector<float>& VEC){
        vector<float> vec(A.get_row(), 0.0), vec_tmp(A.get_row(), 0.0);
        if(A.get_device_mem_stat()){
          util::send(vec, vec_tmp);
        }
        blas::matvec(A, VEC, vec_tmp);
        vml::mul(vec_tmp, 1.0/alpha, vec);
        if(A.get_device_mem_stat()){
          util::device_free(vec_tmp);
        }
        return vec;
      }
    );
  }
  if(A.get_rmatvec_init_flag()){
    C.set_rmatvec(
      [&](const vector<float>& VEC){
        vector<float> vec(A.get_col(), 0.0), vec_tmp(A.get_col(), 0.0);
        if(A.get_device_mem_stat()){
          util::send(vec, vec_tmp);
        }
        blas::rmatvec(A, VEC, vec_tmp);
        vml::mul(vec_tmp, 1.0/alpha, vec);
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
