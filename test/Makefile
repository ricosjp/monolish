all:cpu

cpu:
	make -j -C vector/ $@
	make -j -C matrix/ $@
	make -j -C equation/ $@
	make -j -C logger/ $@
	make -j -C lang/ $@

gpu:
	make -j -C vector/ $@
	make -j -C matrix/ $@
	make -j -C equation/ $@
	make -j -C logger/ $@
	make -j -C lang/ $@

fx:
	make -j -C vector/ $@
	make -j -C matrix/ $@
	make -j -C equation/ $@
	make -j -C logger/ $@
	make -j -C lang/ $@

sx:
	make -j -C vector/ $@
	make -j -C matrix/ $@
	make -j -C equation/ $@
	make -j -C logger/ $@
	make -j -C lang/ $@

run_cpu:
	make -j -C vector/ $@
	make -j -C matrix/ $@
	make -j -C equation/ $@
	make -j -C logger/ $@
	make -j -C lang/ $@

run_gpu:
	make -j -C vector/ $@
	make -j -C matrix/ $@
	make -j -C equation/ $@
	make -j -C logger/ $@
	make -j -C lang/ $@

run_fx:
	make -j -C vector/ $@
	make -j -C matrix/ $@
	make -j -C equation/ $@
	make -j -C logger/ $@
	make -j -C lang/ $@

run_sx:
	make -j -C vector/ $@
	make -j -C matrix/ $@
	make -j -C equation/ $@
	make -j -C logger/ $@
	make -j -C lang/ $@

run_gpu_profile:
	make PROFILER="nsys nvprof" -j -C vector/ run_gpu
	make PROFILER="nsys nvprof" -j -C matrix/ run_gpu
	make PROFILER="nsys nvprof" -j -C equation/ run_gpu
	make PROFILER="nsys nvprof" -j -C logger/ run_gpu
	make PROFILER="nsys nvprof" -j -C lang/ run_gpu
	make -j -C vector/ $@
	make -j -C matrix/ $@
	make -j -C equation/ $@
	make -j -C logger/ $@
	make -j -C lang/ $@

clean:
	make -j -C vector/ clean
	make -j -C matrix/ $@
	make -j -C equation/ clean
	make -j -C logger/ clean
	make -j -C lang/ clean
