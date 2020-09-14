CI_REGISTRY_IMAGE  ?= registry.ritc.jp/ricos/monolish
CI_COMMIT_REF_NAME ?= manual_deploy

.PHONY: test monolish

all: monolish

login:
ifeq ($(CI_BUILD_TOKEN),)
	docker login $(CI_REGISTRY_IMAGE)
else
	docker login -u gitlab-ci-token -p $(CI_BUILD_TOKEN) $(CI_REGISTRY)
endif


monolish: docker/Dockerfile lib/ include/
	docker build -t $(CI_REGISTRY_IMAGE):$(CI_COMMIT_REF_NAME) . -f docker/Dockerfile

push: login monolish
	docker push $(CI_REGISTRY_IMAGE):$(CI_COMMIT_REF_NAME)

in:  
	docker pull  $(CI_REGISTRY_IMAGE):$(CI_COMMIT_REF_NAME)
	docker run -it --gpus all --privileged --mount type=bind,src=$(PWD)/monolish_sample,dst=/test  $(CI_REGISTRY_IMAGE):$(CI_COMMIT_REF_NAME)
	#docker run -it --gpus all --privileged --mount type=bind,src=$(PWD)/test,dst=/test  $(REGISTRY):$(CI_COMMIT_REF_NAME)
