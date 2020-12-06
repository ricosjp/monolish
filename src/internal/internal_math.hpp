void vadd(const size_t N, const double* a, const double* b, double* y, bool gpu_status);
void vsub(const size_t N, const double* a, const double* b, double* y, bool gpu_status);
void vmul(const size_t N, const double* a, const double* b, double* y, bool gpu_status);
void vdiv(const size_t N, const double* a, const double* b, double* y, bool gpu_status);

void vadd(const size_t N, const float* a, const float* b, float* y, bool gpu_status);
void vsub(const size_t N, const float* a, const float* b, float* y, bool gpu_status);
void vmul(const size_t N, const float* a, const float* b, float* y, bool gpu_status);
void vdiv(const size_t N, const float* a, const float* b, float* y, bool gpu_status);
