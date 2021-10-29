#pragma once

namespace monolish {

namespace {
template <typename T>
void ilu_cusolver_get_bufsize(const monolish::matrix::CRS<T> &A, const vector<T> &D,
                        vector<T> &x, const vector<T> &b) {

  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);


  logger.func_out();
}

} // namespace
} // namespace monolish
