#include "../../../include/monolish_vml.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {

//////////////////////////////////////////
// power, sqrt
//////////////////////////////////////////
void vml::pow(const vector<double> &a, const vector<double> &b,
              vector<double> &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(a, b, y));
  assert(util::is_same_device_mem_stat(a, b, y));

  internal::vpow(y.size(), a.data(), b.data(), y.data(),
                 y.get_device_mem_stat());

  logger.func_out();
}

void vml::pow(const vector<double> &a, const double alpha, vector<double> &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(a, y));
  assert(util::is_same_device_mem_stat(a, y));

  internal::vpow(y.size(), a.data(), alpha, y.data(), y.get_device_mem_stat());

  logger.func_out();
}

//////////////////////////////////////////
// other
//////////////////////////////////////////
void vml::max(const vector<double> &a, const vector<double> &b,
              vector<double> &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(a, b, y));
  assert(util::is_same_device_mem_stat(a, b, y));

  internal::vmax(y.size(), a.data(), b.data(), y.data(),
                 y.get_device_mem_stat());

  logger.func_out();
}

double vml::max(const vector<double> &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  double max = internal::vmax(y.size(), y.data(), y.get_device_mem_stat());

  logger.func_out();
  return max;
}

void vml::min(const vector<double> &a, const vector<double> &b,
              vector<double> &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(a, b, y));
  assert(util::is_same_device_mem_stat(a, b, y));

  internal::vmin(y.size(), a.data(), b.data(), y.data(),
                 y.get_device_mem_stat());

  logger.func_out();
}

double vml::min(const vector<double> &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  double min = internal::vmin(y.size(), y.data(), y.get_device_mem_stat());

  logger.func_out();
  return min;
}

} // namespace monolish
