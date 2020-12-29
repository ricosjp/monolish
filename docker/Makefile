CI_REGISTRY_IMAGE  ?= registry.ritc.jp/ricos/monolish
CI_COMMIT_REF_NAME ?= manual_deploy

ALLGEBRA_IMAGE := ghcr.io/ricosjp/allgebra/cuda10_1/clang11gcc7/mkl
ALLGEBRA_TAG   := 20.12.1

MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
MONOLISH_TOP := $(shell dirname ${MAKEFILE_DIR})

.PHONY: allgebra monolish

all: monolish

login:
ifeq ($(CI_BUILD_TOKEN),)
	docker login $(CI_REGISTRY_IMAGE)
else
	docker login -u gitlab-ci-token -p $(CI_BUILD_TOKEN) $(CI_REGISTRY)
endif

monolish: $(MAKEFILE_DIR)/Dockerfile
	docker build -t $(CI_REGISTRY_IMAGE):$(CI_COMMIT_REF_NAME) $(MONOLISH_TOP) -f $(MAKEFILE_DIR)/Dockerfile

push: login monolish
	docker push $(CI_REGISTRY_IMAGE):$(CI_COMMIT_REF_NAME)

in: monolish
	docker run -it --rm --gpus all $(CI_REGISTRY_IMAGE):$(CI_COMMIT_REF_NAME)

test-cpu: monolish
	docker run --rm                 \
		-v $(MONOLISH_TOP)/test:/test \
		-w /test                      \
		$(CI_REGISTRY_IMAGE):$(CI_COMMIT_REF_NAME) bash -c "make cpu; make run_cpu"

test-gpu: monolish
	docker run --rm --gpus all      \
		-v $(MONOLISH_TOP)/test:/test \
		-w /test                      \
		$(CI_REGISTRY_IMAGE):$(CI_COMMIT_REF_NAME) bash -c "make gpu; make run_gpu"
