all:cpu

cpu:
	make -C -j vector_common/ $@
	make -C -j vector_blas/ $@
	make -C -j matrix_common/ $@
	make -C -j matrix_blas/ $@
	make -C equation/ $@
	make -C logger/ $@
	make -C lang/ $@

gpu:
	make -C -j vector_common/ $@
	make -C -j vector_blas/ $@
	make -C -j matrix_common/ $@
	make -C -j matrix_blas/ $@
	make -C equation/ $@
	make -C logger/ $@
	make -C lang/ $@

fx:
	make -C -j vector_common/ $@
	make -C -j vector_blas/ $@
	make -C -j matrix_common/ $@
	make -C -j matrix_blas/ $@
	make -C equation/ $@
	make -C logger/ $@
	make -C lang/ $@

sx:
	make -C -j vector_common/ $@
	make -C -j vector_blas/ $@
	make -C -j matrix_common/ $@
	make -C -j matrix_blas/ $@
	make -C equation/ $@
	make -C logger/ $@
	make -C lang/ $@

run_cpu:
	make -C -j vector_common/ $@
	make -C -j vector_blas/ $@
	make -C -j matrix_common/ $@
	make -C -j matrix_blas/ $@
	make -C equation/ $@
	make -C logger/ $@
	make -C lang/ $@

run_gpu:
	make -C -j vector_common/ $@
	make -C -j vector_blas/ $@
	make -C -j matrix_common/ $@
	make -C -j matrix_blas/ $@
	make -C equation/ $@
	make -C logger/ $@
	make -C lang/ $@

run_fx:
	make -C -j vector_common/ $@
	make -C -j vector_blas/ $@
	make -C -j matrix_common/ $@
	make -C -j matrix_blas/ $@
	make -C equation/ $@
	make -C logger/ $@
	make -C lang/ $@

run_sx:
	make -C -j vector_common/ $@
	make -C -j vector_blas/ $@
	make -C -j matrix_common/ $@
	make -C -j matrix_blas/ $@
	make -C equation/ $@
	make -C logger/ $@
	make -C lang/ $@

run_gpu_profile:
	make PROFILER="nsys nvprof" -C common/ run_gpu
	make PROFILER="nsys nvprof" -C blas/ run_gpu
	make PROFILER="nsys nvprof" -C equation/ run_gpu
	make PROFILER="nsys nvprof" -C logger/ run_gpu
	make PROFILER="nsys nvprof" -C lang/ run_gpu
	make -C -j vector_common/ $@
	make -C -j vector_blas/ $@
	make -C -j matrix_common/ $@
	make -C -j matrix_blas/ $@
	make -C equation/ $@
	make -C logger/ $@
	make -C lang/ $@

clean:
	make -C -j vector_common/ clean
	make -C -j vector_blas/ clean
	make -C -j matrix_common/ clean
	make -C -j matrix_blas/ clean
	make -C equation/ clean
	make -C logger/ clean
	make -C lang/ clean
