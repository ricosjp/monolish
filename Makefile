ALLGEBRA_IMAGE := ghcr.io/ricosjp/allgebra
ALLGEBRA_CUDA := cuda10_1
ALLGEBRA_CC := clang11gcc7
ALLGEBRA_TAG   := 20.12.2

MONOLISH_TOP := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

.PHONY: cpu gpu gcc-cpu clang-cpu clang-gpu test-cpu test-gpu install install-cpu install-gpu in format document

MONOLISH_DIR ?= $(HOME)/lib/monolish

all: cpu gpu

gcc-cpu:
	cmake $(MONOLISH_TOP) \
		-DCMAKE_INSTALL_PREFIX=$(MONOLISH_DIR) \
		-DCMAKE_C_COMPILER=/usr/bin/gcc \
		-DCMAKE_CXX_COMPILER=/usr/bin/g++ \
		-DCMAKE_VERBOSE_MAKEFILE=1 \
		-Bbuild_cpu
	cmake --build build_cpu -j `nproc`

clang-cpu:
	cmake $(MONOLISH_TOP) \
		-DCMAKE_INSTALL_PREFIX=$(MONOLISH_DIR) \
		-DCMAKE_C_COMPILER=/usr/local/llvm-11.0.0/bin/clang \
		-DCMAKE_C_COMPILER=/usr/local/llvm-11.0.0/bin/clang++ \
		-DCMAKE_VERBOSE_MAKEFILE=1 \
		-Bbuild_gpu \
	cmake --build build_gpu -j `nproc`

clang-gpu:
	cmake $(MONOLISH_TOP) \
		-DCMAKE_INSTALL_PREFIX=$(MONOLISH_DIR) \
		-DCMAKE_C_COMPILER=/usr/local/llvm-11.0.0/bin/clang \
		-DCMAKE_C_COMPILER=/usr/local/llvm-11.0.0/bin/clang++ \
		-DCMAKE_VERBOSE_MAKEFILE=1 \
		-Bbuild_gpu \
		-DMONOLISH_USE_GPU=ON
	cmake --build build_gpu -j `nproc`

cpu: gcc-cpu
gpu: clang-gpu

a64fx:
	$(MAKE) -B -j4 -f Makefile.a64fx

sxat:
	$(MAKE) -B -j -f Makefile.sxat

install-cpu: cpu
	cmake --build build_cpu --target install

install-gpu: gpu
	cmake --build build_gpu --target install

install-sxat: 
	$(MAKE) -B -j -f Makefile.sxat install

install-a64fx: 
	$(MAKE) -B -j -f Makefile.a64fx install


install: install-cpu install-gpu

test-cpu: install-cpu
	$(MAKE) -C test cpu
	$(MAKE) -C test run_cpu

test-gpu: install-gpu
	$(MAKE) -C test gpu
	$(MAKE) -C test run_gpu

test:
	test-cpu
	test-gpu

clean:
	rm -rf build*/
	$(MAKE) -f Makefile.a64fx clean
	$(MAKE) -f Makefile.sxat clean
	$(MAKE) -C test/ clean

in-mkl-gpu:
	docker run -it --rm \
		--gpus all   \
		-e MONOLISH_DIR=/opt/monolish/0.1 \
		-e LD_LIBRARY_PATH=/opt/monolish/0.1/lib \
		-v $(MONOLISH_TOP):/monolish \
		-w /monolish \
		$(ALLGEBRA_IMAGE)/$(ALLGEBRA_CUDA)/$(ALLGEBRA_CC)/mkl:$(ALLGEBRA_TAG)

in-mkl-cpu:
	docker run -it --rm \
		-e MONOLISH_DIR=/opt/monolish/0.1 \
		-e LD_LIBRARY_PATH=/opt/monolish/0.1/lib \
		-v $(MONOLISH_TOP):/monolish \
		-w /monolish \
		$(ALLGEBRA_IMAGE)/$(ALLGEBRA_CUDA)/$(ALLGEBRA_CC)/mkl:$(ALLGEBRA_TAG)

in-oss-gpu:
	docker run -it --rm \
		--gpus all   \
		-e MONOLISH_DIR=/opt/monolish/0.1 \
		-e LD_LIBRARY_PATH=/opt/monolish/0.1/lib \
		-v $(MONOLISH_TOP):/monolish \
		-w /monolish \
		$(ALLGEBRA_IMAGE)/$(ALLGEBRA_CUDA)/$(ALLGEBRA_CC)/oss:$(ALLGEBRA_TAG)

in-oss-cpu:
	docker run -it --rm \
		-e MONOLISH_DIR=/opt/monolish/0.1 \
		-e LD_LIBRARY_PATH=/opt/monolish/0.1/lib \
		-v $(MONOLISH_TOP):/monolish \
		-w /monolish \
		$(ALLGEBRA_IMAGE)/$(ALLGEBRA_CUDA)/$(ALLGEBRA_CC)/oss:$(ALLGEBRA_TAG)

in-cpu: in-mkl-cpu
in-gpu: in-mkl-gpu
in: in-gpu

format:
	docker run -it --rm  \
		-u `id -u`:`id -g` \
		-v $(PWD):$(PWD)   \
		-w $(PWD)          \
		$(ALLGEBRA_IMAGE)/clang-format:20.10.1 /usr/bin/check-format.sh

document:
	docker run -it --rm  \
		-u `id -u`:`id -g` \
		-v $(PWD):$(PWD)   \
		-w $(PWD)          \
		$(ALLGEBRA_IMAGE)/doxygen:20.10.1 doxygen Doxyfile
