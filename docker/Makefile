REGISTRY  := registry.ritc.jp/ricos/allgebra/monolish
CI_COMMIT_REF_NAME ?= manual_deploy
.PHONY: test allgebra 


all: allgebra

login:
ifeq ($(CI_BUILD_TOKEN),)
	docker login $(REGISTRY)
else
	docker login -u gitlab-ci-token -p $(CI_BUILD_TOKEN) $(REGISTRY)
endif

allgebra: Dockerfile lib/ include/
	docker build -t $(REGISTRY):aaa . -f Dockerfile
	#docker build -t $(REGISTRY):$(CI_COMMIT_REF_NAME) . -f Dockerfile

push: login allgebra
	docker push $(REGISTRY):aaa

in:  
	docker run -it --privileged --mount type=bind,src=$(PWD)/test,dst=/test  $(REGISTRY):aaa
	#docker run -it --gpus all --privileged --mount type=bind,src=$(PWD)/test,dst=/test  $(REGISTRY):$(CI_COMMIT_REF_NAME)
