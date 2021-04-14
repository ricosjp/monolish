REGISTRY := ghcr.io/ricosjp/monolish
TAG := 0.14.0

MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
MONOLISH_TOP := $(shell dirname ${MAKEFILE_DIR})

.PHONY: allgebra monolish oss mkl oss_nvidia mkl_nvidia

all: monolish

login:
ifeq ($(CI_BUILD_TOKEN),)
	docker login $(REGISTRY)
else
	docker login -u gitlab-ci-token -p $(CI_BUILD_TOKEN) $(CI_REGISTRY)
endif

monolish: oss mkl oss_nvidia mkl_nvidia

oss: oss.Dockerfile
	cp -r ../examples ./monolish_examples
	docker build -t $(REGISTRY)/$@:$(TAG) . -f $<
	rm -rf ./monolish_examples/ 

mkl: $(MAKEFILE_DIR)/mkl.Dockerfile
	cp -r ../examples ./monolish_examples
	docker build -t $(REGISTRY)/$@:$(TAG) . -f $<
	rm -rf ./monolish_examples/ 

oss_nvidia: $(MAKEFILE_DIR)/oss_nvidia.Dockerfile
	cp -r ../examples ./monolish_examples
	docker build -t $(REGISTRY)/$@:$(TAG) . -f $<
	rm -rf ./monolish_examples/ 

mkl_nvidia: $(MAKEFILE_DIR)/mkl_nvidia.Dockerfile
	cp -r ../examples ./monolish_examples
	docker build -t $(REGISTRY)/$@:$(TAG) . -f $<
	rm -rf ./monolish_examples/ 

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

test-cpu: monolish
	docker run --rm                 \
		-v $(MONOLISH_TOP)/test:/test \
		-w /test                      \
		$(REGISTRY):$(TAG) bash -c "make cpu; make run_cpu"

test-gpu: monolish
	docker run --rm --gpus all      \
		-v $(MONOLISH_TOP)/test:/test \
		-w /test                      \
		$(REGISTRY):$(TAG) bash -c "make gpu; make run_gpu"
