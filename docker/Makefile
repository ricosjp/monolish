REGISTRY := ghcr.io/ricosjp/monolish
TAG := 0.14.2
#TAG := latest

MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
MONOLISH_TOP := $(shell dirname ${MAKEFILE_DIR})

.PHONY: allgebra monolish oss mkl oss_nvidia mkl_nvidia test_oss test_mkl test_oss_nvidia test_mkl_nvidia

all: monolish

login:
ifeq ($(CI_BUILD_TOKEN),)
	docker login $(REGISTRY)
else
	docker login -u hishinuma -p ${GITHUB_TOKEN_HISHINUMA} $(REGISTRY)
endif

monolish: oss mkl oss_nvidia mkl_nvidia

oss: oss.Dockerfile
	cp -r ../examples ./monolish_examples
	cp -r ../benchmark ./monolish_benchmark
	docker build -t $(REGISTRY)/$@:$(TAG) . -f $<
	rm -rf ./monolish_examples/  monolish_benchmark/

mkl: $(MAKEFILE_DIR)/mkl.Dockerfile
	cp -r ../examples ./monolish_examples
	cp -r ../benchmark ./monolish_benchmark
	docker build -t $(REGISTRY)/$@:$(TAG) . -f $<
	rm -rf ./monolish_examples/  monolish_benchmark/

oss_nvidia: $(MAKEFILE_DIR)/oss_nvidia.Dockerfile
	cp -r ../examples ./monolish_examples
	cp -r ../benchmark ./monolish_benchmark
	docker build -t $(REGISTRY)/$@:$(TAG) . -f $<
	rm -rf ./monolish_examples/  monolish_benchmark/

mkl_nvidia: $(MAKEFILE_DIR)/mkl_nvidia.Dockerfile
	cp -r ../examples ./monolish_examples
	cp -r ../benchmark ./monolish_benchmark
	docker build -t $(REGISTRY)/$@:$(TAG) . -f $<
	rm -rf ./monolish_examples/  monolish_benchmark/

push: login monolish
	docker push $(REGISTRY)/oss:$(TAG)
	docker push $(REGISTRY)/mkl:$(TAG)
	docker push $(REGISTRY)/oss_nvidia:$(TAG)
	docker push $(REGISTRY)/mkl_nvidia:$(TAG)

in_oss: oss
	docker run -it --rm $(REGISTRY)/$<:$(TAG)

in_mkl: mkl
	docker run -it --rm $(REGISTRY)/$<:$(TAG)

in_oss_nvidia: oss_nvidia
	docker run -it --rm --gpus all $(REGISTRY)/$<:$(TAG)

in_mkl_nvidia: mkl_nvidia
	docker run -it --rm --gpus all $(REGISTRY)/$<:$(TAG)

test: test_oss test_mkl test_oss_nvidia test_mkl_nvidia

test_oss:
	docker run --rm                 \
		-w /opt/monolish/examples/blas/innerproduct \
		$(REGISTRY)/oss:$(TAG) bash -c "make cpu; ./innerproduct_cpu.out"

test_mkl:
	docker run --rm                 \
		-w /opt/monolish/examples/blas/innerproduct \
		$(REGISTRY)/mkl:$(TAG) bash -c "make cpu; ./innerproduct_cpu.out"

test_oss_nvidia:
	docker run --rm --gpus all                \
		-w /opt/monolish/examples/blas/innerproduct \
		$(REGISTRY)/oss_nvidia:$(TAG) bash -c "sh /opt/monolish/link_monolish_gpu.sh; make gpu; ./innerproduct_gpu.out"

test_mkl_nvidia:
	docker run --rm --gpus all                \
		-w /opt/monolish/examples/blas/innerproduct \
		$(REGISTRY)/mkl_nvidia:$(TAG) bash -c "sh /opt/monolish/link_monolish_gpu.sh; make gpu; ./innerproduct_gpu.out"
