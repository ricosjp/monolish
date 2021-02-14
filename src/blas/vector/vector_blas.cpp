
#include "../../../include/monolish_blas.hpp"
#include "../../internal/monolish_internal.hpp"
#include "asum.hpp"
#include "axpy.hpp"
#include "axpyz.hpp"
#include "copy.hpp"
#include "dot.hpp"
#include "nrm1.hpp"
#include "nrm2.hpp"
#include "scal.hpp"
#include "sum.hpp"
#include "vecadd.hpp"
#include "vecsub.hpp"
#include "xpay.hpp"

namespace monolish {
namespace blas {

void vecadd(const vector<double> &a, const vector<double> &b, vector<double> &y){vecadd_core(a, b, y);}
void vecadd(const vector<double> &a, const vector<double> &b, view1D<vector<double>,double> &y){vecadd_core(a, b, y);}
void vecadd(const vector<double> &a, const vector<double> &b, view1D<matrix::Dense<double>,double> &y){vecadd_core(a, b, y);}
void vecadd(const vector<double> &a, const view1D<vector<double>,double> &b, vector<double> &y){vecadd_core(a, b, y);}
void vecadd(const vector<double> &a, const view1D<vector<double>,double> &b, view1D<vector<double>,double> &y){vecadd_core(a, b, y);}
void vecadd(const vector<double> &a, const view1D<vector<double>,double> &b, view1D<matrix::Dense<double>,double> &y){vecadd_core(a, b, y);}
void vecadd(const vector<double> &a, const view1D<matrix::Dense<double>,double> &b, vector<double> &y){vecadd_core(a, b, y);}
void vecadd(const vector<double> &a, const view1D<matrix::Dense<double>,double> &b, view1D<vector<double>,double> &y){vecadd_core(a, b, y);}
void vecadd(const vector<double> &a, const view1D<matrix::Dense<double>,double> &b, view1D<matrix::Dense<double>,double> &y){vecadd_core(a, b, y);}
void vecadd(const view1D<vector<double>,double> &a, const vector<double> &b, vector<double> &y){vecadd_core(a, b, y);}
void vecadd(const view1D<vector<double>,double> &a, const vector<double> &b, view1D<vector<double>,double> &y){vecadd_core(a, b, y);}
void vecadd(const view1D<vector<double>,double> &a, const vector<double> &b, view1D<matrix::Dense<double>,double> &y){vecadd_core(a, b, y);}
void vecadd(const view1D<vector<double>,double> &a, const view1D<vector<double>,double> &b, vector<double> &y){vecadd_core(a, b, y);}
void vecadd(const view1D<vector<double>,double> &a, const view1D<vector<double>,double> &b, view1D<vector<double>,double> &y){vecadd_core(a, b, y);}
void vecadd(const view1D<vector<double>,double> &a, const view1D<vector<double>,double> &b, view1D<matrix::Dense<double>,double> &y){vecadd_core(a, b, y);}
void vecadd(const view1D<vector<double>,double> &a, const view1D<matrix::Dense<double>,double> &b, vector<double> &y){vecadd_core(a, b, y);}
void vecadd(const view1D<vector<double>,double> &a, const view1D<matrix::Dense<double>,double> &b, view1D<vector<double>,double> &y){vecadd_core(a, b, y);}
void vecadd(const view1D<vector<double>,double> &a, const view1D<matrix::Dense<double>,double> &b, view1D<matrix::Dense<double>,double> &y){vecadd_core(a, b, y);}
void vecadd(const view1D<matrix::Dense<double>,double> &a, const vector<double> &b, vector<double> &y){vecadd_core(a, b, y);}
void vecadd(const view1D<matrix::Dense<double>,double> &a, const vector<double> &b, view1D<vector<double>,double> &y){vecadd_core(a, b, y);}
void vecadd(const view1D<matrix::Dense<double>,double> &a, const vector<double> &b, view1D<matrix::Dense<double>,double> &y){vecadd_core(a, b, y);}
void vecadd(const view1D<matrix::Dense<double>,double> &a, const view1D<vector<double>,double> &b, vector<double> &y){vecadd_core(a, b, y);}
void vecadd(const view1D<matrix::Dense<double>,double> &a, const view1D<vector<double>,double> &b, view1D<vector<double>,double> &y){vecadd_core(a, b, y);}
void vecadd(const view1D<matrix::Dense<double>,double> &a, const view1D<vector<double>,double> &b, view1D<matrix::Dense<double>,double> &y){vecadd_core(a, b, y);}
void vecadd(const view1D<matrix::Dense<double>,double> &a, const view1D<matrix::Dense<double>,double> &b, vector<double> &y){vecadd_core(a, b, y);}
void vecadd(const view1D<matrix::Dense<double>,double> &a, const view1D<matrix::Dense<double>,double> &b, view1D<vector<double>,double> &y){vecadd_core(a, b, y);}
void vecadd(const view1D<matrix::Dense<double>,double> &a, const view1D<matrix::Dense<double>,double> &b, view1D<matrix::Dense<double>,double> &y){vecadd_core(a, b, y);}
void vecadd(const vector<float> &a, const vector<float> &b, vector<float> &y){vecadd_core(a, b, y);}
void vecadd(const vector<float> &a, const vector<float> &b, view1D<vector<float>,float> &y){vecadd_core(a, b, y);}
void vecadd(const vector<float> &a, const vector<float> &b, view1D<matrix::Dense<float>,float> &y){vecadd_core(a, b, y);}
void vecadd(const vector<float> &a, const view1D<vector<float>,float> &b, vector<float> &y){vecadd_core(a, b, y);}
void vecadd(const vector<float> &a, const view1D<vector<float>,float> &b, view1D<vector<float>,float> &y){vecadd_core(a, b, y);}
void vecadd(const vector<float> &a, const view1D<vector<float>,float> &b, view1D<matrix::Dense<float>,float> &y){vecadd_core(a, b, y);}
void vecadd(const vector<float> &a, const view1D<matrix::Dense<float>,float> &b, vector<float> &y){vecadd_core(a, b, y);}
void vecadd(const vector<float> &a, const view1D<matrix::Dense<float>,float> &b, view1D<vector<float>,float> &y){vecadd_core(a, b, y);}
void vecadd(const vector<float> &a, const view1D<matrix::Dense<float>,float> &b, view1D<matrix::Dense<float>,float> &y){vecadd_core(a, b, y);}
void vecadd(const view1D<vector<float>,float> &a, const vector<float> &b, vector<float> &y){vecadd_core(a, b, y);}
void vecadd(const view1D<vector<float>,float> &a, const vector<float> &b, view1D<vector<float>,float> &y){vecadd_core(a, b, y);}
void vecadd(const view1D<vector<float>,float> &a, const vector<float> &b, view1D<matrix::Dense<float>,float> &y){vecadd_core(a, b, y);}
void vecadd(const view1D<vector<float>,float> &a, const view1D<vector<float>,float> &b, vector<float> &y){vecadd_core(a, b, y);}
void vecadd(const view1D<vector<float>,float> &a, const view1D<vector<float>,float> &b, view1D<vector<float>,float> &y){vecadd_core(a, b, y);}
void vecadd(const view1D<vector<float>,float> &a, const view1D<vector<float>,float> &b, view1D<matrix::Dense<float>,float> &y){vecadd_core(a, b, y);}
void vecadd(const view1D<vector<float>,float> &a, const view1D<matrix::Dense<float>,float> &b, vector<float> &y){vecadd_core(a, b, y);}
void vecadd(const view1D<vector<float>,float> &a, const view1D<matrix::Dense<float>,float> &b, view1D<vector<float>,float> &y){vecadd_core(a, b, y);}
void vecadd(const view1D<vector<float>,float> &a, const view1D<matrix::Dense<float>,float> &b, view1D<matrix::Dense<float>,float> &y){vecadd_core(a, b, y);}
void vecadd(const view1D<matrix::Dense<float>,float> &a, const vector<float> &b, vector<float> &y){vecadd_core(a, b, y);}
void vecadd(const view1D<matrix::Dense<float>,float> &a, const vector<float> &b, view1D<vector<float>,float> &y){vecadd_core(a, b, y);}
void vecadd(const view1D<matrix::Dense<float>,float> &a, const vector<float> &b, view1D<matrix::Dense<float>,float> &y){vecadd_core(a, b, y);}
void vecadd(const view1D<matrix::Dense<float>,float> &a, const view1D<vector<float>,float> &b, vector<float> &y){vecadd_core(a, b, y);}
void vecadd(const view1D<matrix::Dense<float>,float> &a, const view1D<vector<float>,float> &b, view1D<vector<float>,float> &y){vecadd_core(a, b, y);}
void vecadd(const view1D<matrix::Dense<float>,float> &a, const view1D<vector<float>,float> &b, view1D<matrix::Dense<float>,float> &y){vecadd_core(a, b, y);}
void vecadd(const view1D<matrix::Dense<float>,float> &a, const view1D<matrix::Dense<float>,float> &b, vector<float> &y){vecadd_core(a, b, y);}
void vecadd(const view1D<matrix::Dense<float>,float> &a, const view1D<matrix::Dense<float>,float> &b, view1D<vector<float>,float> &y){vecadd_core(a, b, y);}
void vecadd(const view1D<matrix::Dense<float>,float> &a, const view1D<matrix::Dense<float>,float> &b, view1D<matrix::Dense<float>,float> &y){vecadd_core(a, b, y);}

void vecsub(const vector<double> &a, const vector<double> &b, vector<double> &y){vecsub_core(a, b, y);}
void vecsub(const vector<double> &a, const vector<double> &b, view1D<vector<double>,double> &y){vecsub_core(a, b, y);}
void vecsub(const vector<double> &a, const vector<double> &b, view1D<matrix::Dense<double>,double> &y){vecsub_core(a, b, y);}
void vecsub(const vector<double> &a, const view1D<vector<double>,double> &b, vector<double> &y){vecsub_core(a, b, y);}
void vecsub(const vector<double> &a, const view1D<vector<double>,double> &b, view1D<vector<double>,double> &y){vecsub_core(a, b, y);}
void vecsub(const vector<double> &a, const view1D<vector<double>,double> &b, view1D<matrix::Dense<double>,double> &y){vecsub_core(a, b, y);}
void vecsub(const vector<double> &a, const view1D<matrix::Dense<double>,double> &b, vector<double> &y){vecsub_core(a, b, y);}
void vecsub(const vector<double> &a, const view1D<matrix::Dense<double>,double> &b, view1D<vector<double>,double> &y){vecsub_core(a, b, y);}
void vecsub(const vector<double> &a, const view1D<matrix::Dense<double>,double> &b, view1D<matrix::Dense<double>,double> &y){vecsub_core(a, b, y);}
void vecsub(const view1D<vector<double>,double> &a, const vector<double> &b, vector<double> &y){vecsub_core(a, b, y);}
void vecsub(const view1D<vector<double>,double> &a, const vector<double> &b, view1D<vector<double>,double> &y){vecsub_core(a, b, y);}
void vecsub(const view1D<vector<double>,double> &a, const vector<double> &b, view1D<matrix::Dense<double>,double> &y){vecsub_core(a, b, y);}
void vecsub(const view1D<vector<double>,double> &a, const view1D<vector<double>,double> &b, vector<double> &y){vecsub_core(a, b, y);}
void vecsub(const view1D<vector<double>,double> &a, const view1D<vector<double>,double> &b, view1D<vector<double>,double> &y){vecsub_core(a, b, y);}
void vecsub(const view1D<vector<double>,double> &a, const view1D<vector<double>,double> &b, view1D<matrix::Dense<double>,double> &y){vecsub_core(a, b, y);}
void vecsub(const view1D<vector<double>,double> &a, const view1D<matrix::Dense<double>,double> &b, vector<double> &y){vecsub_core(a, b, y);}
void vecsub(const view1D<vector<double>,double> &a, const view1D<matrix::Dense<double>,double> &b, view1D<vector<double>,double> &y){vecsub_core(a, b, y);}
void vecsub(const view1D<vector<double>,double> &a, const view1D<matrix::Dense<double>,double> &b, view1D<matrix::Dense<double>,double> &y){vecsub_core(a, b, y);}
void vecsub(const view1D<matrix::Dense<double>,double> &a, const vector<double> &b, vector<double> &y){vecsub_core(a, b, y);}
void vecsub(const view1D<matrix::Dense<double>,double> &a, const vector<double> &b, view1D<vector<double>,double> &y){vecsub_core(a, b, y);}
void vecsub(const view1D<matrix::Dense<double>,double> &a, const vector<double> &b, view1D<matrix::Dense<double>,double> &y){vecsub_core(a, b, y);}
void vecsub(const view1D<matrix::Dense<double>,double> &a, const view1D<vector<double>,double> &b, vector<double> &y){vecsub_core(a, b, y);}
void vecsub(const view1D<matrix::Dense<double>,double> &a, const view1D<vector<double>,double> &b, view1D<vector<double>,double> &y){vecsub_core(a, b, y);}
void vecsub(const view1D<matrix::Dense<double>,double> &a, const view1D<vector<double>,double> &b, view1D<matrix::Dense<double>,double> &y){vecsub_core(a, b, y);}
void vecsub(const view1D<matrix::Dense<double>,double> &a, const view1D<matrix::Dense<double>,double> &b, vector<double> &y){vecsub_core(a, b, y);}
void vecsub(const view1D<matrix::Dense<double>,double> &a, const view1D<matrix::Dense<double>,double> &b, view1D<vector<double>,double> &y){vecsub_core(a, b, y);}
void vecsub(const view1D<matrix::Dense<double>,double> &a, const view1D<matrix::Dense<double>,double> &b, view1D<matrix::Dense<double>,double> &y){vecsub_core(a, b, y);}
void vecsub(const vector<float> &a, const vector<float> &b, vector<float> &y){vecsub_core(a, b, y);}
void vecsub(const vector<float> &a, const vector<float> &b, view1D<vector<float>,float> &y){vecsub_core(a, b, y);}
void vecsub(const vector<float> &a, const vector<float> &b, view1D<matrix::Dense<float>,float> &y){vecsub_core(a, b, y);}
void vecsub(const vector<float> &a, const view1D<vector<float>,float> &b, vector<float> &y){vecsub_core(a, b, y);}
void vecsub(const vector<float> &a, const view1D<vector<float>,float> &b, view1D<vector<float>,float> &y){vecsub_core(a, b, y);}
void vecsub(const vector<float> &a, const view1D<vector<float>,float> &b, view1D<matrix::Dense<float>,float> &y){vecsub_core(a, b, y);}
void vecsub(const vector<float> &a, const view1D<matrix::Dense<float>,float> &b, vector<float> &y){vecsub_core(a, b, y);}
void vecsub(const vector<float> &a, const view1D<matrix::Dense<float>,float> &b, view1D<vector<float>,float> &y){vecsub_core(a, b, y);}
void vecsub(const vector<float> &a, const view1D<matrix::Dense<float>,float> &b, view1D<matrix::Dense<float>,float> &y){vecsub_core(a, b, y);}
void vecsub(const view1D<vector<float>,float> &a, const vector<float> &b, vector<float> &y){vecsub_core(a, b, y);}
void vecsub(const view1D<vector<float>,float> &a, const vector<float> &b, view1D<vector<float>,float> &y){vecsub_core(a, b, y);}
void vecsub(const view1D<vector<float>,float> &a, const vector<float> &b, view1D<matrix::Dense<float>,float> &y){vecsub_core(a, b, y);}
void vecsub(const view1D<vector<float>,float> &a, const view1D<vector<float>,float> &b, vector<float> &y){vecsub_core(a, b, y);}
void vecsub(const view1D<vector<float>,float> &a, const view1D<vector<float>,float> &b, view1D<vector<float>,float> &y){vecsub_core(a, b, y);}
void vecsub(const view1D<vector<float>,float> &a, const view1D<vector<float>,float> &b, view1D<matrix::Dense<float>,float> &y){vecsub_core(a, b, y);}
void vecsub(const view1D<vector<float>,float> &a, const view1D<matrix::Dense<float>,float> &b, vector<float> &y){vecsub_core(a, b, y);}
void vecsub(const view1D<vector<float>,float> &a, const view1D<matrix::Dense<float>,float> &b, view1D<vector<float>,float> &y){vecsub_core(a, b, y);}
void vecsub(const view1D<vector<float>,float> &a, const view1D<matrix::Dense<float>,float> &b, view1D<matrix::Dense<float>,float> &y){vecsub_core(a, b, y);}
void vecsub(const view1D<matrix::Dense<float>,float> &a, const vector<float> &b, vector<float> &y){vecsub_core(a, b, y);}
void vecsub(const view1D<matrix::Dense<float>,float> &a, const vector<float> &b, view1D<vector<float>,float> &y){vecsub_core(a, b, y);}
void vecsub(const view1D<matrix::Dense<float>,float> &a, const vector<float> &b, view1D<matrix::Dense<float>,float> &y){vecsub_core(a, b, y);}
void vecsub(const view1D<matrix::Dense<float>,float> &a, const view1D<vector<float>,float> &b, vector<float> &y){vecsub_core(a, b, y);}
void vecsub(const view1D<matrix::Dense<float>,float> &a, const view1D<vector<float>,float> &b, view1D<vector<float>,float> &y){vecsub_core(a, b, y);}
void vecsub(const view1D<matrix::Dense<float>,float> &a, const view1D<vector<float>,float> &b, view1D<matrix::Dense<float>,float> &y){vecsub_core(a, b, y);}
void vecsub(const view1D<matrix::Dense<float>,float> &a, const view1D<matrix::Dense<float>,float> &b, vector<float> &y){vecsub_core(a, b, y);}
void vecsub(const view1D<matrix::Dense<float>,float> &a, const view1D<matrix::Dense<float>,float> &b, view1D<vector<float>,float> &y){vecsub_core(a, b, y);}
void vecsub(const view1D<matrix::Dense<float>,float> &a, const view1D<matrix::Dense<float>,float> &b, view1D<matrix::Dense<float>,float> &y){vecsub_core(a, b, y);}

void copy(const vector<double> &x, vector<double> &y){copy_core(x, y);}
void copy(const vector<double> &x, view1D<vector<double>,double> &y){copy_core(x, y);}
void copy(const vector<double> &x, view1D<matrix::Dense<double>,double> &y){copy_core(x, y);}
void copy(const view1D<vector<double>,double> &x, vector<double> &y){copy_core(x, y);}
void copy(const view1D<vector<double>,double> &x, view1D<vector<double>,double> &y){copy_core(x, y);}
void copy(const view1D<vector<double>,double> &x, view1D<matrix::Dense<double>,double> &y){copy_core(x, y);}
void copy(const view1D<matrix::Dense<double>,double> &x, vector<double> &y){copy_core(x, y);}
void copy(const view1D<matrix::Dense<double>,double> &x, view1D<vector<double>,double> &y){copy_core(x, y);}
void copy(const view1D<matrix::Dense<double>,double> &x, view1D<matrix::Dense<double>,double> &y){copy_core(x, y);}
void copy(const vector<float> &x, vector<float> &y){copy_core(x, y);}
void copy(const vector<float> &x, view1D<vector<float>,float> &y){copy_core(x, y);}
void copy(const vector<float> &x, view1D<matrix::Dense<float>,float> &y){copy_core(x, y);}
void copy(const view1D<vector<float>,float> &x, vector<float> &y){copy_core(x, y);}
void copy(const view1D<vector<float>,float> &x, view1D<vector<float>,float> &y){copy_core(x, y);}
void copy(const view1D<vector<float>,float> &x, view1D<matrix::Dense<float>,float> &y){copy_core(x, y);}
void copy(const view1D<matrix::Dense<float>,float> &x, vector<float> &y){copy_core(x, y);}
void copy(const view1D<matrix::Dense<float>,float> &x, view1D<vector<float>,float> &y){copy_core(x, y);}
void copy(const view1D<matrix::Dense<float>,float> &x, view1D<matrix::Dense<float>,float> &y){copy_core(x, y);}

double asum(const vector<double> &x){ return Dasum_core(x); }
double asum(const view1D<vector<double>,double> &x){ return Dasum_core(x); }
double asum(const view1D<matrix::Dense<double>,double> &x){ return Dasum_core(x); }
float asum(const vector<float> &x){ return Sasum_core(x); }
float asum(const view1D<vector<float>,float> &x){ return Sasum_core(x); }
float asum(const view1D<matrix::Dense<float>,float> &x){ return Sasum_core(x); }

void asum(const vector<double> &x, double &ans){ ans = asum(x); }
void asum(const view1D<vector<double>,double> &x, double &ans){ ans = asum(x); }
void asum(const view1D<matrix::Dense<double>,double> &x, double &ans){ ans = asum(x); }
void asum(const vector<float> &x, float &ans){ ans = asum(x); }
void asum(const view1D<vector<float>,float> &x, float &ans){ ans = asum(x); }
void asum(const view1D<matrix::Dense<float>,float> &x, float &ans){ ans = asum(x); }

double sum(const vector<double> &x){ return Dsum_core(x); }
double sum(const view1D<vector<double>,double> &x){ return Dsum_core(x); }
double sum(const view1D<matrix::Dense<double>,double> &x){ return Dsum_core(x); }
float sum(const vector<float> &x){ return Ssum_core(x); }
float sum(const view1D<vector<float>,float> &x){ return Ssum_core(x); }
float sum(const view1D<matrix::Dense<float>,float> &x){ return Ssum_core(x); }

void sum(const vector<double> &x, double &ans){ ans = sum(x); }
void sum(const view1D<vector<double>,double> &x, double &ans){ ans = sum(x); }
void sum(const view1D<matrix::Dense<double>,double> &x, double &ans){ ans = sum(x); }
void sum(const vector<float> &x, float &ans){ ans = sum(x); }
void sum(const view1D<vector<float>,float> &x, float &ans){ ans = sum(x); }
void sum(const view1D<matrix::Dense<float>,float> &x, float &ans){ ans = sum(x); }

} // namespace blas
} // namespace monolish
