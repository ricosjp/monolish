ALLGEBRA_IMAGE := ghcr.io/ricosjp/allgebra/cuda10_2/mkl
ALLGEBRA_TAG   := 20.10.0

MONOLISH_TOP := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

.PHONY: cpu gpu gpu-debug lib test install in format

MONOLISH_DIR ?= /opt/monolish

all:cpu gpu

cpu:
	cmake . \
		-DCMAKE_INSTALL_PREFIX=$(MONOLISH_DIR) \
		-Bbuild_cpu
	cmake --build build_cpu -j `nproc`

cpu-debug:
	cmake . \
		-DCMAKE_INSTALL_PREFIX=$(MONOLISH_DIR) \
		-DCMAKE_BUILD_TYPE=Debug \
		-Bbuild_cpu_debug
	cmake --build build_cpu_debug -j `nproc`

gpu:
	cmake . \
		-DCMAKE_INSTALL_PREFIX=$(MONOLISH_DIR) \
		-Bbuild_gpu \
		-DBUILD_GPU=ON
	cmake --build build_gpu -j `nproc`

gpu-debug:
	cmake . \
		-DCMAKE_INSTALL_PREFIX=$(MONOLISH_DIR) \
		-DCMAKE_BUILD_TYPE=Debug \
		-Bbuild_gpu_debug \
		-DBUILD_GPU=ON
	cmake --build build_gpu_debug -j `nproc`

fx:
	make -B -j4 -f Makefile.fx

sx:
	make -B -j -f Makefile.sx

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

test:
	cd test; make -B

clean:
	- rm -rf build*/
	- make -f Makefile.fx clean
	- make -f Makefile.sx clean
	- make -C test/ clean

zenbu:
	make clean
	make cpu
	make gpu
	make install

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
