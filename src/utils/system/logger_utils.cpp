#include "../../../include/monolish_blas.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {

// solver
void Logger::solver_in(std::string func_name) {
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

void Logger::solver_out() {
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

// func
void Logger::func_in(std::string func_name) {
  if (LogLevel >= 2) {
    if (filename.empty()) {
      pStream = &std::cout;
    }

    calls.push_back(func_name);
    times.push_back(std::chrono::system_clock::now());
  }
}

void Logger::func_out() {
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

// util
void Logger::util_in(std::string func_name) {
  if (LogLevel >= 3) {
    if (filename.empty()) {
      pStream = &std::cout;
    }
    calls.push_back(func_name);
    times.push_back(std::chrono::system_clock::now());
  }
}

void Logger::util_out() {
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

/// logger util ///
void util::set_log_level(size_t Level) {
  Logger &logger = Logger::get_instance();
  logger.set_log_level(Level);
}

void util::set_log_filename(std::string filename) {
  Logger &logger = Logger::get_instance();
  logger.set_log_filename(filename);
}

} // namespace monolish
