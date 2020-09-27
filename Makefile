ALLGEBRA_IMAGE := ghcr.io/ricosjp/allgebra/cuda10_2/mkl
ALLGEBRA_TAG   := 20.10.0

MONOLISH_TOP := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

.PHONY: cpu-debug gpu gpu-debug test-cpu test-gpu install install-cpu install-gpu in format document

MONOLISH_DIR ?= $(HOME)/lib/monolish

all: cpu gpu

cpu:
	cmake $(MONOLISH_TOP) \
		-DCMAKE_INSTALL_PREFIX=$(MONOLISH_DIR) \
		-DCMAKE_VERBOSE_MAKEFILE=1 \
		-Bbuild_cpu
	cmake --build build_cpu -j `nproc`

cpu-debug:
	cmake $(MONOLISH_TOP) \
		-DCMAKE_INSTALL_PREFIX=$(MONOLISH_DIR) \
		-DCMAKE_VERBOSE_MAKEFILE=1 \
		-DCMAKE_BUILD_TYPE=Debug \
		-Bbuild_cpu_debug
	cmake --build build_cpu_debug -j `nproc`

gpu:
	cmake $(MONOLISH_TOP) \
		-DCMAKE_INSTALL_PREFIX=$(MONOLISH_DIR) \
		-DCMAKE_VERBOSE_MAKEFILE=1 \
		-Bbuild_gpu \
		-DBUILD_GPU=ON
	cmake --build build_gpu -j `nproc`

gpu-debug:
	cmake $(MONOLISH_TOP) \
		-DCMAKE_INSTALL_PREFIX=$(MONOLISH_DIR) \
		-DCMAKE_VERBOSE_MAKEFILE=1 \
		-DCMAKE_BUILD_TYPE=Debug \
		-Bbuild_gpu_debug \
		-DBUILD_GPU=ON
	cmake --build build_gpu_debug -j `nproc`

fx:
	$(MAKE) -B -j4 -f Makefile.fx

sx:
	$(MAKE) -B -j -f Makefile.sx

install-cpu: cpu
	cmake --build build_cpu --target install

install-cpu-debug: cpu-debug
	cmake --build build_cpu_debug --target install

install-gpu: gpu
	cmake --build build_gpu --target install

install-gpu-debug: gpu-debug
	cmake --build build_gpu_debug --target install

install: install-cpu install-gpu
install-debug: install-cpu-debug install-gpu-debug

test-cpu: install-cpu
	$(MAKE) -C test cpu
	$(MAKE) -C test run_cpu

test-gpu: install-gpu
	$(MAKE) -C test gpu
	$(MAKE) -C test run_gpu

clean:
	rm -rf build*/
	$(MAKE) -f Makefile.fx clean
	$(MAKE) -f Makefile.sx clean
	$(MAKE) -C test/ clean

in:
	docker run -it --rm \
		--gpus all   \
		-e MONOLISH_DIR=/opt/monolish/0.1 \
		-e LD_LIBRARY_PATH=/opt/monolish/0.1/lib \
		-v $(MONOLISH_TOP):/monolish \
		-w /monolish \
		$(ALLGEBRA_IMAGE):$(ALLGEBRA_TAG)

in-cpu:
	docker run -it --rm \
		-e MONOLISH_DIR=/opt/monolish/0.1 \
		-e LD_LIBRARY_PATH=/opt/monolish/0.1/lib \
		-v $(MONOLISH_TOP):/monolish \
		-w /monolish \
		$(ALLGEBRA_IMAGE):$(ALLGEBRA_TAG)

format:
	docker run -it --rm  \
		-u `id -u`:`id -g` \
		-v $(PWD):$(PWD)   \
		-w $(PWD)          \
		ghcr.io/ricosjp/allgebra/clang-format:20.10.0 /usr/bin/check-format.sh

document:
	docker run -it --rm  \
		-u `id -u`:`id -g` \
		-v $(PWD):$(PWD)   \
		-w $(PWD)          \
		ghcr.io/ricosjp/allgebra/doxygen:20.10.0 doxygen Doxyfile
