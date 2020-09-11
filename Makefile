.ONESHELL:

SHELL := /bin/bash
DATE_ID := $(shell date +"%y.%m.%d")

# Get package name from pwd
SOURCE_DIR = source /opt/intel/openvino/bin/setupvars.sh

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT


.PHONY: clean clean-test clean-pyc clean-build help changelog run

help:
	@python3 -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

changelog: ## Generate changelog for current repo
	docker run -it --rm -v "$(pwd)":/usr/local/src/your-app \
	ferrarimarco/github-changelog-generator -u "$(USER)" -p face_mask_detection_openvino

formatter: ## Format style with black
	isort -rc .
	black -l 90 .

lint: ## check style with flake8
	flake8 --max-line-length 90 .

example: ## Run main.py with example
	xhost +;
	docker run --rm -ti --volume "$(CURDIR)":/app --env DISPLAY=$(DISPLAY) \
	--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --device /dev/snd \
	--device /dev/video0 mmphego/intel-openvino \
	bash -c "source /opt/intel/openvino/bin/setupvars.sh && \
		python main.py --face-model models/face-detection-adas-0001 \
		--mask-model models/face_mask \
		-i resources/mask.mp4 \
		--debug \
		--show-bbox \
		--enable-speech"

