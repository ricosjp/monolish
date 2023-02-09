#pragma once
#include <chrono>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <string>
#include <vector>

#define monolish_func __FUNCTION__

#if defined USE_MPI
#include <mpi.h>
#endif

namespace monolish {
/**
 * @brief logger class (singleton, for developper class)
 */
class Logger {
private:
  Logger() = default;

  ~Logger() {
    if (pStream != &std::cout) {
      delete pStream;
    }
  };

  std::vector<std::string> calls;
  std::vector<std::chrono::system_clock::time_point> times;
  std::string filename;
  std::ostream *pStream;

public:
  size_t LogLevel = 0;

  Logger(const Logger &) = delete;
  Logger &operator=(const Logger &) = delete;
  Logger(Logger &&) = delete;
  Logger &operator=(Logger &&) = delete;

  [[nodiscard]] static Logger &get_instance() {
    static Logger instance;
    return instance;
  }

  /**
   * @brief Specifying the log level
   * @param L loglevel
   * @note loglevel is
   * 1. logging solvers (CG, Jacobi, LU...etc.)
   * 2. logging solvers and BLAS functions (matmul, matvec, arithmetic
   *operators..etc.)
   * 3. logging solvers and BLAS functions and utils (send, recv,
   *allocation...etc.)
   * @details see also monolish::util::set_log_level()
   **/
  void set_log_level(size_t L) {
    if (3 < L) { // loglevel = {0, 1, 2, 3}
      throw std::runtime_error("error bad LogLevel");
    }
    LogLevel = L;
  }

  /**
   * @brief Specifying the log finename
   * @param file the log filename
   * @details see also monolish::util::set_log_filename()
   **/
  void set_log_filename(const std::string file) {
    filename = file;

    // file open
    pStream = new std::ofstream(filename);
    if (pStream->fail()) {
      delete pStream;
      pStream = &std::cout;
    }
  }

  // for solver (large func.)
  void solver_in(const std::string func_name);
  void solver_out();

  // for blas (small func.)
  void func_in(const std::string func_name);
  void func_out();

  // for utils (very small func.)
  void util_in(const std::string func_name);
  void util_out();
};
} // namespace monolish
