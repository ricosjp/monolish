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
  double stime;
  double etime;
  std::string filename;
  std::ostream *pStream;

public:
  size_t LogLevel = 0;

  Logger(const Logger &) = delete;
  Logger &operator=(const Logger &) = delete;
  Logger(Logger &&) = delete;
  Logger &operator=(Logger &&) = delete;

  static Logger &get_instance() {
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
  void set_log_filename(std::string file) {
    filename = file;

    // file open
    pStream = new std::ofstream(filename);
    if (pStream->fail()) {
      delete pStream;
      pStream = &std::cout;
    }
  }

  // for solver (large func.)
  void solver_in(std::string func_name) {
    if (LogLevel >= 1) {
      if (filename.empty()) {
        pStream = &std::cout;
      }

      // init
      calls.push_back(func_name);
      times.push_back(std::chrono::system_clock::now());

      // start
      *pStream << "- {" << std::flush;

      // func
      *pStream << "type : solver, " << std::flush;
      *pStream << "name : " << std::flush;
      for (int i = 0; i < (int)calls.size(); i++)
        *pStream << calls[i] << "/" << std::flush;
      *pStream << ", " << std::flush;

      // stat
      *pStream << "stat : IN" << std::flush;

      // end
      *pStream << "}" << std::endl;
    }
  }

  void solver_out() {
    if (LogLevel >= 1) {
      if (filename.empty()) {
        pStream = &std::cout;
      }

      // start
      *pStream << "- {" << std::flush;

      // func
      *pStream << "type : solver, " << std::flush;
      *pStream << "name : " << std::flush;
      for (int i = 0; i < (int)calls.size(); i++)
        *pStream << calls[i] << "/" << std::flush;
      *pStream << ", " << std::flush;

      // time
      auto end = std::chrono::system_clock::now();
      double sec = std::chrono::duration_cast<std::chrono::nanoseconds>(
                       end - times[(int)times.size() - 1])
                       .count() /
                   1.0e+9;
      *pStream << "stat : OUT, " << std::flush;
      *pStream << "time : " << sec << std::flush;

      // end
      *pStream << "}" << std::endl;

      calls.pop_back();
      times.pop_back();
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////

  // for blas (small func.)
  void func_in(std::string func_name) {
    if (LogLevel >= 2) {
      if (filename.empty()) {
        pStream = &std::cout;
      }

      calls.push_back(func_name);
      times.push_back(std::chrono::system_clock::now());
    }
  }

  void func_out() {
    if (LogLevel >= 2) {
      if (filename.empty()) {
        pStream = &std::cout;
      }

      // start
      *pStream << "- {" << std::flush;

      // func
      *pStream << "type : func, " << std::flush;
      *pStream << "name : " << std::flush;
      for (int i = 0; i < (int)calls.size(); i++)
        *pStream << calls[i] << "/" << std::flush;
      *pStream << ", " << std::flush;

      // time
      auto end = std::chrono::system_clock::now();
      double sec = std::chrono::duration_cast<std::chrono::nanoseconds>(
                       end - times[(int)times.size() - 1])
                       .count() /
                   1.0e+9;
      *pStream << "time : " << sec << std::flush;

      // end
      *pStream << "}" << std::endl;

      calls.pop_back();
      times.pop_back();
    }
  }

  // for utils (very small func.)
  void util_in(std::string func_name) {
    if (LogLevel >= 3) {
      if (filename.empty()) {
        pStream = &std::cout;
      }
      calls.push_back(func_name);
      times.push_back(std::chrono::system_clock::now());
    }
  }

  void util_out() {
    if (LogLevel >= 3) {
      if (filename.empty()) {
        pStream = &std::cout;
      }

      // start
      *pStream << "- {" << std::flush;

      // func
      *pStream << "type : util, " << std::flush;
      *pStream << "name : " << std::flush;
      for (int i = 0; i < (int)calls.size(); i++)
        *pStream << calls[i] << "/" << std::flush;
      *pStream << ", " << std::flush;

      // time
      auto end = std::chrono::system_clock::now();
      double sec = std::chrono::duration_cast<std::chrono::nanoseconds>(
                       end - times[(int)times.size() - 1])
                       .count() /
                   1.0e+9;
      *pStream << "time : " << sec << std::flush;

      // end
      *pStream << "}" << std::endl;

      calls.pop_back();
      times.pop_back();
    }
  }
};
} // namespace monolish
