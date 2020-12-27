#include "../../include/monolish_blas.hpp"
#include "../internal/monolish_internal.hpp"

namespace monolish {

bool util::solver_check(const int err) {
  switch (err) {
  case MONOLISH_SOLVER_SUCCESS:
    return 0;
  case MONOLISH_SOLVER_MAXITER:
    std::runtime_error("equation error, maxiter\n");
    return false;
  case MONOLISH_SOLVER_BREAKDOWN:
    std::runtime_error("equation error, breakdown\n");
    return false;
  case MONOLISH_SOLVER_SIZE_ERROR:
    std::runtime_error("equation error, size error\n");
    return false;
  case MONOLISH_SOLVER_RESIDUAL_NAN:
    std::runtime_error("equation error, resudual is nan\n");
    return false;
  case MONOLISH_SOLVER_NOT_IMPL:
    std::runtime_error("equation error, this solver is not impl.\n");
    return false;
  default:
    return 0;
  }
}

} // namespace monolish
