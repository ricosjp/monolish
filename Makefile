ALLGEBRA_IMAGE := ghcr.io/ricosjp/allgebra/cuda10_2/mkl
ALLGEBRA_TAG   := 20.10.0

MONOLISH_TOP := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

.PHONY: gpu test-cpu test-gpu install install-cpu install-gpu in format document

MONOLISH_DIR ?= $(HOME)/lib/monolish

all: cpu gpu

cpu:
	cmake $(MONOLISH_TOP) \
		-DCMAKE_INSTALL_PREFIX=$(MONOLISH_DIR) \
		-DCMAKE_VERBOSE_MAKEFILE=1 \
		-Bbuild_cpu
	cmake --build build_cpu -j `nproc`

gpu:
	cmake $(MONOLISH_TOP) \
		-DCMAKE_INSTALL_PREFIX=$(MONOLISH_DIR) \
		-DCMAKE_VERBOSE_MAKEFILE=1 \
		-Bbuild_gpu \
		-DMONOLISH_USE_GPU=ON
	cmake --build build_gpu -j `nproc`

fx:
	$(MAKE) -B -j4 -f Makefile.fx

sx:
	$(MAKE) -B -j -f Makefile.sx

install-cpu: cpu
	cmake --build build_cpu --target install

install-gpu: gpu
	cmake --build build_gpu --target install

install: install-cpu install-gpu

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
