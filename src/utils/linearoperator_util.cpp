#include "../../include/common/monolish_logger.hpp"
#include "../../include/common/monolish_matrix.hpp"
#include "../../include/common/monolish_vector.hpp"
#include "../internal/monolish_internal.hpp"
#include"../../include/monolish_blas.hpp"

#include <cassert>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

namespace monolish {
namespace matrix {

// constructor //
template <typename T>
LinearOperator<T>::LinearOperator(const size_t M, const size_t N){
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  rowN = M;
  colN = N;
  gpu_status = false;
  matvec_init_flag = false;
  rmatvec_init_flag = false;
  logger.util_out();
}
template LinearOperator<double>::LinearOperator(const size_t M, const size_t N);
template LinearOperator<float>::LinearOperator(const size_t M, const size_t N);

template <typename T>
LinearOperator<T>::LinearOperator(const size_t M, const size_t N, const std::function<vector<T>(const vector<T>&)>& MATVEC){
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  rowN = M;
  colN = N;
  gpu_status = false;
  matvec = MATVEC;
  matvec_init_flag = true;
  rmatvec_init_flag = false;
  logger.util_out();
}

template LinearOperator<double>::LinearOperator(const size_t M, const size_t N, const std::function<vector<double>(const vector<double>&)>& MATVEC);
template LinearOperator<float>::LinearOperator(const size_t M, const size_t N, const std::function<vector<float>(const vector<float>&)>& MATVEC);

template <typename T>
LinearOperator<T>::LinearOperator(const size_t M, const size_t N, const std::function<vector<T>(const vector<T>&)>& MATVEC, const std::function<vector<T>(const vector<T>&)>& RMATVEC){
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  rowN = M;
  colN = N;
  gpu_status = false;
  matvec = MATVEC;
  matvec_init_flag = true;
  rmatvec = RMATVEC;
  rmatvec_init_flag = true;
  logger.util_out();
}

template LinearOperator<double>::LinearOperator(const size_t M, const size_t N, const std::function<vector<double>(const vector<double>&)>& MATVEC, const std::function<vector<double>(const vector<double>&)>& RMATVEC);
template LinearOperator<float>::LinearOperator(const size_t M, const size_t N, const std::function<vector<float>(const vector<float>&)>& MATVEC, const std::function<vector<float>(const vector<float>&)>& RMATVEC);

// copy constructor
template <typename T> LinearOperator<T>::LinearOperator(const LinearOperator<T>& linearoperator){
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  rowN = linearoperator.get_row();
  colN = linearoperator.get_col();

  gpu_status = linearoperator.get_device_mem_stat();

  matvec = linearoperator.get_matvec();
  matvec_init_flag = linearoperator.get_matvec_init_flag();

  rmatvec = linearoperator.get_rmatvec();
  rmatvec_init_flag = linearoperator.get_rmatvec_init_flag();
}

template LinearOperator<double>::LinearOperator(const LinearOperator<double> &linearoperator);
template LinearOperator<float>::LinearOperator(const LinearOperator<float> &linearoperator);

template <typename T> void LinearOperator<T>::convert(COO<T> &coo){
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  // todo coo err check (only square)

  rowN = coo.get_row();
  colN = coo.get_col();

  gpu_status = coo.get_device_mem_stat();

  set_matvec([&](const monolish::vector<T>& VEC){
    CRS<T> crs(coo);
    monolish::vector<T> vec(crs.get_row(), 0);
    monolish::blas::matvec(crs, VEC, vec);
    return vec;
  });
  rmatvec_init_flag = false;

  logger.util_out();
}

template void LinearOperator<double>::convert(COO<double> &coo);
template void LinearOperator<float>::convert(COO<float> &coo);

template <typename T> void LinearOperator<T>::convert(CRS<T> &crs){
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  // todo crs err check (only square)

  rowN = crs.get_row();
  colN = crs.get_col();

  gpu_status = crs.get_device_mem_stat();

  set_matvec([&](const monolish::vector<T>& VEC){
    monolish::vector<T> vec(crs.get_row(), 0);
    if(gpu_status) {
      monolish::util::send(vec);
    }
    monolish::blas::matvec(crs, VEC, vec);
    return vec;
  });
  rmatvec_init_flag = false;

  logger.util_out();
}

template void LinearOperator<double>::convert(CRS<double> &crs);
template void LinearOperator<float>::convert(CRS<float> &crs);

template <typename T>
void LinearOperator<T>::set_matvec(const std::function<vector<T>(const vector<T>&)>& MATVEC){
  matvec = MATVEC;
  matvec_init_flag = true;
}

template void LinearOperator<double>::set_matvec(const std::function<vector<double>(const vector<double>&)>& MATVEC);
template void LinearOperator<float>::set_matvec(const std::function<vector<float>(const vector<float>&)>& MATVEC);

template <typename T>
void LinearOperator<T>::set_rmatvec(const std::function<vector<T>(const vector<T>&)>& RMATVEC){
  rmatvec = RMATVEC;
  rmatvec_init_flag = true;
}

template void LinearOperator<double>::set_rmatvec(const std::function<vector<double>(const vector<double>&)>& RMATVEC);
template void LinearOperator<float>::set_rmatvec(const std::function<vector<float>(const vector<float>&)>& RMATVEC);

template <typename T>
void LinearOperator<T>::convert_to_Dense(Dense<T>& dense) const {
  if(!matvec_init_flag){
    Dense<T> A(rowN, colN);
    dense = A;
    return;
  }

  std::vector<T> values(rowN*colN);
  for(size_t i = 0; i < colN; ++i) {
    std::vector<T> vec_tmp(colN, 0);
    vec_tmp[i] = 1;
    vector<T> vec(vec_tmp);
    vector<T> ans(rowN);
    if(gpu_status){
      util::send(ans, vec);
    }
    fprintf(stderr, "check3-1\n");
    ans = matvec(vec);
    fprintf(stderr, "check3-2\n");
    if(gpu_status){
      util::recv(ans);
    }
    fprintf(stderr, "check3-3\n");
    for(size_t j = 0; j < rowN; ++j){
      values[j*colN+i] = ans[j];
    }
  }

  dense = Dense<T>(rowN, colN, values);
}

template void LinearOperator<double>::convert_to_Dense(Dense<double>&) const ;
template void LinearOperator<float>::convert_to_Dense(Dense<float>&) const ;

} // namespace matrix
} // namespace monolish
