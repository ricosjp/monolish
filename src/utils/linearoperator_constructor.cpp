#include "../../include/common/monolish_logger.hpp"
#include "../../include/common/monolish_matrix.hpp"
#include "../../include/common/monolish_vector.hpp"
#include "../../include/monolish_blas.hpp"
#include "../internal/monolish_internal.hpp"

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
  rmatvec = RMATVEC;
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

template <typename T>
LinearOperator<T>::LinearOperator(
    const size_t M, const size_t N,
    const std::function<Dense<T>(const Dense<T> &)> &MATMUL) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  rowN = M;
  colN = N;
  gpu_status = false;
  matmul_dense = MATMUL;
  logger.util_out();
}

template LinearOperator<double>::LinearOperator(
    const size_t M, const size_t N,
    const std::function<Dense<double>(const Dense<double> &)> &MATMUL);
template LinearOperator<float>::LinearOperator(
    const size_t M, const size_t N,
    const std::function<Dense<float>(const Dense<float> &)> &MATMUL);

template <typename T>
LinearOperator<T>::LinearOperator(
    const size_t M, const size_t N,
    const std::function<Dense<T>(const Dense<T> &)> &MATMUL,
    const std::function<Dense<T>(const Dense<T> &)> &RMATMUL) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  rowN = M;
  colN = N;
  gpu_status = false;
  matmul_dense = MATMUL;
  rmatmul_dense = RMATMUL;
  logger.util_out();
}

template LinearOperator<double>::LinearOperator(
    const size_t M, const size_t N,
    const std::function<Dense<double>(const Dense<double> &)> &MATMUL,
    const std::function<Dense<double>(const Dense<double> &)> &RMATMUL);
template LinearOperator<float>::LinearOperator(
    const size_t M, const size_t N,
    const std::function<Dense<float>(const Dense<float> &)> &MATMUL,
    const std::function<Dense<float>(const Dense<float> &)> &RMATMUL);

// copy constructor
template <typename T>
LinearOperator<T>::LinearOperator(const LinearOperator<T> &linearoperator) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  rowN = linearoperator.get_row();
  colN = linearoperator.get_col();

  gpu_status = linearoperator.get_device_mem_stat();

  matvec = linearoperator.get_matvec();
  rmatvec = linearoperator.get_rmatvec();
}

template LinearOperator<double>::LinearOperator(
    const LinearOperator<double> &linearoperator);
template LinearOperator<float>::LinearOperator(
    const LinearOperator<float> &linearoperator);

template <typename T>
void LinearOperator<T>::set_matvec(
    const std::function<vector<T>(const vector<T> &)> &MATVEC) {
  matvec = MATVEC;
}

template void LinearOperator<double>::set_matvec(
    const std::function<vector<double>(const vector<double> &)> &MATVEC);
template void LinearOperator<float>::set_matvec(
    const std::function<vector<float>(const vector<float> &)> &MATVEC);

template <typename T>
void LinearOperator<T>::set_rmatvec(
    const std::function<vector<T>(const vector<T> &)> &RMATVEC) {
  rmatvec = RMATVEC;
}

template void LinearOperator<double>::set_rmatvec(
    const std::function<vector<double>(const vector<double> &)> &RMATVEC);
template void LinearOperator<float>::set_rmatvec(
    const std::function<vector<float>(const vector<float> &)> &RMATVEC);

template <typename T>
void LinearOperator<T>::set_matmul_dense(
    const std::function<Dense<T>(const Dense<T> &)> &MATMUL) {
  matmul_dense = MATMUL;
}

template void LinearOperator<double>::set_matmul_dense(
    const std::function<Dense<double>(const Dense<double> &)> &MATMUL);
template void LinearOperator<float>::set_matmul_dense(
    const std::function<Dense<float>(const Dense<float> &)> &MATMUL);

template <typename T>
void LinearOperator<T>::set_rmatmul_dense(
    const std::function<Dense<T>(const Dense<T> &)> &RMATMUL) {
  rmatmul_dense = RMATMUL;
}

template void LinearOperator<double>::set_rmatmul_dense(
    const std::function<Dense<double>(const Dense<double> &)> &RMATMUL);
template void LinearOperator<float>::set_rmatmul_dense(
    const std::function<Dense<float>(const Dense<float> &)> &RMATMUL);

} // namespace matrix
} // namespace monolish
