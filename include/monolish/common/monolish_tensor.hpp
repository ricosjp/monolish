#pragma once

namespace monolish {
/**
 * @brief Declare dense tensor class
 */

namespace tensor {
template <typename Float> class tensor_Dense;
}
} // namespace monolish

#include "./monolish_coo.hpp"
#include "./monolish_crs.hpp"
#include "./monolish_dense.hpp"
#include "./monolish_linearoperator.hpp"
#include "./monolish_tensor_coo.hpp"
#include "./monolish_tensor_crs.hpp"
#include "./monolish_tensor_dense.hpp"
