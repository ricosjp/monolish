CI_REGISTRY_IMAGE  ?= registry.ritc.jp/ricos/monolish
CI_COMMIT_REF_NAME ?= manual_deploy

ALLGEBRA_IMAGE := registry.ritc.jp/ricos/allgebra/cuda10_2
ALLGEBRA_TAG   := latest

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

allgebra:
	docker pull $(ALLGEBRA_IMAGE):$(ALLGEBRA_TAG)
	docker run -it --rm \
		--gpus all   \
		--privileged \
		-e MONOLISH_DIR=/opt/monolish/0.1 \
		-v $(MONOLISH_TOP):/monolish \
		-w /monolish \
		$(ALLGEBRA_IMAGE):$(ALLGEBRA_TAG)

in: monolish
	docker run -it \
		--gpus all   \
		--privileged \
		-v $(MAKEFILE_DIR)/monolish_sample:/test \
		$(CI_REGISTRY_IMAGE):$(CI_COMMIT_REF_NAME)
