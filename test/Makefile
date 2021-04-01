all:cpu

cpu:
	make -j -C vector/ $@
	make -j -C view1D/ $@
	make -j -C matrix/ $@
	make -j -C equation/ $@
	make -j -C eigen/ $@
	make -j -C logger/ $@
	make -j -C lang/ $@

gpu:
	make -j -C vector/ $@
	make -j -C view1D/ $@
	make -j -C matrix/ $@
	make -j -C equation/ $@
	make -j -C eigen/ $@
	make -j -C logger/ $@
	make -j -C lang/ $@

a64fx:
	make -j -C vector/ $@
	make -j -C view1D/ $@
	make -j -C matrix/ $@
	make -j -C equation/ $@
	make -j -C eigen/ $@
	make -j -C logger/ $@
	make -j -C lang/ $@

sxat:
	make -j -C vector/ $@
	make -j -C view1D/ $@
	make -j -C matrix/ $@
	make -j -C equation/ $@
	make -j -C eigen/ $@
	make -j -C logger/ $@
	make -j -C lang/ $@

run_cpu:
	make -j -C vector/ $@
	make -j -C view1D/ $@
	make -j -C matrix/ $@
	make -j -C equation/ $@
	make -j -C eigen/ $@
	make -j -C logger/ $@
	make -j -C lang/ $@

run_gpu:
	make -j -C vector/ $@
	make -j -C view1D/ $@
	make -j -C matrix/ $@
	make -j -C equation/ $@
	make -j -C eigen/ $@
	make -j -C logger/ $@
	make -j -C lang/ $@

run_a64fx:
	make -j -C vector/ $@
	make -j -C view1D/ $@
	make -j -C matrix/ $@
	make -j -C equation/ $@
	make -j -C eigen/ $@
	make -j -C logger/ $@
	make -j -C lang/ $@

run_sxat:
	make -j -C vector/ $@
	make -j -C view1D/ $@
	make -j -C matrix/ $@
	make -j -C equation/ $@
	make -j -C eigen/ $@
	make -j -C logger/ $@
	make -j -C lang/ $@

run_gpu_profile:
	make PROFILER="nsys nvprof" -j -C vector/ run_gpu
	make PROFILER="nsys nvprof" -j -C matrix/ run_gpu
	make PROFILER="nsys nvprof" -j -C equation/ run_gpu
	make PROFILER="nsys nvprof" -j -C eigen/ run_gpu
	make PROFILER="nsys nvprof" -j -C logger/ run_gpu
	make PROFILER="nsys nvprof" -j -C lang/ run_gpu
	make -j -C vector/ $@
	make -j -C matrix/ $@
	make -j -C equation/ $@
	make -j -C eigen/ $@
	make -j -C logger/ $@
	make -j -C lang/ $@

clean:
	make -j -C vector/ clean
	make -j -C view1D/ $@
	make -j -C matrix/ $@
	make -j -C equation/ clean
	make -j -C eigen/ $@
	make -j -C logger/ clean
	make -j -C lang/ clean
