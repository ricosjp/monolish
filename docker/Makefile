CI_REGISTRY_IMAGE  ?= registry.ritc.jp/ricos/monolish
CI_COMMIT_REF_NAME ?= manual_deploy

####
CI_COMMIT_REF_NAME = 34-monolish-docker-image
####
.PHONY: test allgebra 

all: allgebra

login:
ifeq ($(CI_BUILD_TOKEN),)
	docker login $(CI_REGISTRY)
else
	docker login -u gitlab-ci-token -p $(CI_BUILD_TOKEN) $(CI_REGISTRY)
endif


allgebra: Dockerfile lib/ include/
	docker build -t $(CI_REGISTRY_IMAGE):$(CI_COMMIT_REF_NAME) . -f Dockerfile

push: login allgebra
	docker push $(CI_REGISTRY_IMAGE):$(CI_COMMIT_REF_NAME)

in:  
	docker pull  $(CI_REGISTRY_IMAGE):$(CI_COMMIT_REF_NAME)
	docker run -it --gpus all --privileged --mount type=bind,src=$(PWD)/test,dst=/test  $(CI_REGISTRY_IMAGE):$(CI_COMMIT_REF_NAME)
	#docker run -it --gpus all --privileged --mount type=bind,src=$(PWD)/test,dst=/test  $(REGISTRY):$(CI_COMMIT_REF_NAME)
