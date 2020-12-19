#include "../../include/monolish_vml.hpp"
#include "../internal/monolish_internal.hpp"

namespace monolish {

//////////////////////////////////////////
// power, sqrt
//////////////////////////////////////////
void vml::pow(const vector<float> &a, const vector<float> &b,
              vector<float> &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (a.size() != b.size() || a.size() != y.size()) {
    throw std::runtime_error("error vector size is not same");
  }
  if (a.get_device_mem_stat() != b.get_device_mem_stat() ||
      a.get_device_mem_stat() != y.get_device_mem_stat()) {
    throw std::runtime_error("error vector get_device_mem_stat() is not same");
  }

  internal::vpow(y.size(), a.data(), b.data(), y.data(),
                 y.get_device_mem_stat());

  logger.func_out();
}

void vml::pow(const vector<float> &a, const float alpha, vector<float> &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (a.size() != y.size()) {
    throw std::runtime_error("error vector size is not same");
  }
  if (a.get_device_mem_stat() != y.get_device_mem_stat()) {
    throw std::runtime_error("error vector get_device_mem_stat() is not same");
  }

  internal::vpow(y.size(), a.data(), alpha, y.data(), y.get_device_mem_stat());

  logger.func_out();
}

void vml::sqrt(const vector<float> &a, vector<float> &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (a.size() != y.size()) {
    throw std::runtime_error("error vector size is not same");
  }
  if (a.get_device_mem_stat() != y.get_device_mem_stat()) {
    throw std::runtime_error("error vector get_device_mem_stat() is not same");
  }

  internal::vsqrt(y.size(), a.data(), y.data(), y.get_device_mem_stat());

  logger.func_out();
}

//////////////////////////////////////////
// sin
//////////////////////////////////////////
void vml::sin(const vector<float> &a, vector<float> &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (a.size() != y.size()) {
    throw std::runtime_error("error vector size is not same");
  }
  if (a.get_device_mem_stat() != y.get_device_mem_stat()) {
    throw std::runtime_error("error vector get_device_mem_stat() is not same");
  }

  internal::vsin(y.size(), a.data(), y.data(), y.get_device_mem_stat());

  logger.func_out();
}

void vml::sinh(const vector<float> &a, vector<float> &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (a.size() != y.size()) {
    throw std::runtime_error("error vector size is not same");
  }
  if (a.get_device_mem_stat() != y.get_device_mem_stat()) {
    throw std::runtime_error("error vector get_device_mem_stat() is not same");
  }

  internal::vsinh(y.size(), a.data(), y.data(), y.get_device_mem_stat());

  logger.func_out();
}

void vml::asin(const vector<float> &a, vector<float> &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (a.size() != y.size()) {
    throw std::runtime_error("error vector size is not same");
  }
  if (a.get_device_mem_stat() != y.get_device_mem_stat()) {
    throw std::runtime_error("error vector get_device_mem_stat() is not same");
  }

  internal::vasin(y.size(), a.data(), y.data(), y.get_device_mem_stat());

  logger.func_out();
}

void vml::asinh(const vector<float> &a, vector<float> &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (a.size() != y.size()) {
    throw std::runtime_error("error vector size is not same");
  }
  if (a.get_device_mem_stat() != y.get_device_mem_stat()) {
    throw std::runtime_error("error vector get_device_mem_stat() is not same");
  }

  internal::vasinh(y.size(), a.data(), y.data(), y.get_device_mem_stat());

  logger.func_out();
}

//////////////////////////////////////////
// tan
//////////////////////////////////////////
void vml::tan(const vector<float> &a, vector<float> &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (a.size() != y.size()) {
    throw std::runtime_error("error vector size is not same");
  }
  if (a.get_device_mem_stat() != y.get_device_mem_stat()) {
    throw std::runtime_error("error vector get_device_mem_stat() is not same");
  }

  internal::vtan(y.size(), a.data(), y.data(), y.get_device_mem_stat());

  logger.func_out();
}

void vml::tanh(const vector<float> &a, vector<float> &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (a.size() != y.size()) {
    throw std::runtime_error("error vector size is not same");
  }
  if (a.get_device_mem_stat() != y.get_device_mem_stat()) {
    throw std::runtime_error("error vector get_device_mem_stat() is not same");
  }

  internal::vtanh(y.size(), a.data(), y.data(), y.get_device_mem_stat());

  logger.func_out();
}

void vml::atan(const vector<float> &a, vector<float> &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (a.size() != y.size()) {
    throw std::runtime_error("error vector size is not same");
  }
  if (a.get_device_mem_stat() != y.get_device_mem_stat()) {
    throw std::runtime_error("error vector get_device_mem_stat() is not same");
  }

  internal::vatan(y.size(), a.data(), y.data(), y.get_device_mem_stat());

  logger.func_out();
}

void vml::atanh(const vector<float> &a, vector<float> &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (a.size() != y.size()) {
    throw std::runtime_error("error vector size is not same");
  }
  if (a.get_device_mem_stat() != y.get_device_mem_stat()) {
    throw std::runtime_error("error vector get_device_mem_stat() is not same");
  }

  internal::vatanh(y.size(), a.data(), y.data(), y.get_device_mem_stat());

  logger.func_out();
}

//////////////////////////////////////////
// other
//////////////////////////////////////////

void vml::ceil(const vector<float> &a, vector<float> &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (a.size() != y.size()) {
    throw std::runtime_error("error vector size is not same");
  }
  if (a.get_device_mem_stat() != y.get_device_mem_stat()) {
    throw std::runtime_error("error vector get_device_mem_stat() is not same");
  }

  internal::vceil(y.size(), a.data(), y.data(), y.get_device_mem_stat());

  logger.func_out();
}

void vml::floor(const vector<float> &a, vector<float> &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (a.size() != y.size()) {
    throw std::runtime_error("error vector size is not same");
  }
  if (a.get_device_mem_stat() != y.get_device_mem_stat()) {
    throw std::runtime_error("error vector get_device_mem_stat() is not same");
  }

  internal::vfloor(y.size(), a.data(), y.data(), y.get_device_mem_stat());

  logger.func_out();
}

void vml::sign(const vector<float> &a, vector<float> &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (a.size() != y.size()) {
    throw std::runtime_error("error vector size is not same");
  }
  if (a.get_device_mem_stat() != y.get_device_mem_stat()) {
    throw std::runtime_error("error vector get_device_mem_stat() is not same");
  }

  internal::vsign(y.size(), a.data(), y.data(), y.get_device_mem_stat());

  logger.func_out();
}

} // namespace monolish
