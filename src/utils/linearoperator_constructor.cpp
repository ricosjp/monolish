#include "../../include/common/monolish_logger.hpp"
#include "../../include/common/monolish_matrix.hpp"
#include "../../include/common/monolish_vector.hpp"
#include "../../include/monolish_blas.hpp"
#include "../internal/monolish_internal.hpp"

#include <cassert>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

namespace monolish {
namespace matrix {

// constructor //
template <typename T>
LinearOperator<T>::LinearOperator(const size_t M, const size_t N) {
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
LinearOperator<T>::LinearOperator(
    const size_t M, const size_t N,
    const std::function<vector<T>(const vector<T> &)> &MATVEC) {
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

template LinearOperator<double>::LinearOperator(
    const size_t M, const size_t N,
    const std::function<vector<double>(const vector<double> &)> &MATVEC);
template LinearOperator<float>::LinearOperator(
    const size_t M, const size_t N,
    const std::function<vector<float>(const vector<float> &)> &MATVEC);

template <typename T>
LinearOperator<T>::LinearOperator(
    const size_t M, const size_t N,
    const std::function<vector<T>(const vector<T> &)> &MATVEC,
    const std::function<vector<T>(const vector<T> &)> &RMATVEC) {
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

template LinearOperator<double>::LinearOperator(
    const size_t M, const size_t N,
    const std::function<vector<double>(const vector<double> &)> &MATVEC,
    const std::function<vector<double>(const vector<double> &)> &RMATVEC);
template LinearOperator<float>::LinearOperator(
    const size_t M, const size_t N,
    const std::function<vector<float>(const vector<float> &)> &MATVEC,
    const std::function<vector<float>(const vector<float> &)> &RMATVEC);

// copy constructor
template <typename T>
LinearOperator<T>::LinearOperator(const LinearOperator<T> &linearoperator) {
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

template LinearOperator<double>::LinearOperator(
    const LinearOperator<double> &linearoperator);
template LinearOperator<float>::LinearOperator(
    const LinearOperator<float> &linearoperator);

template <typename T>
void LinearOperator<T>::set_matvec(
    const std::function<vector<T>(const vector<T> &)> &MATVEC) {
  matvec = MATVEC;
  matvec_init_flag = true;
}

template void LinearOperator<double>::set_matvec(
    const std::function<vector<double>(const vector<double> &)> &MATVEC);
template void LinearOperator<float>::set_matvec(
    const std::function<vector<float>(const vector<float> &)> &MATVEC);

template <typename T>
void LinearOperator<T>::set_rmatvec(
    const std::function<vector<T>(const vector<T> &)> &RMATVEC) {
  rmatvec = RMATVEC;
  rmatvec_init_flag = true;
}

template void LinearOperator<double>::set_rmatvec(
    const std::function<vector<double>(const vector<double> &)> &RMATVEC);
template void LinearOperator<float>::set_rmatvec(
    const std::function<vector<float>(const vector<float> &)> &RMATVEC);

} // namespace matrix
} // namespace monolish
