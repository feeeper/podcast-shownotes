clear:
	rm -rf ./.mypy_cache ./.pytest_cache
	find . -type f -name "*.pyc" -print0 | xargs -r0 rm

format:
	$(CONDA_RUN) python -m brunette src tests \
      --single-quotes \
      --line-length 79 \
      -t py310

lint: flake8 mypy

# minus to ignore exit code and proceed to next targets
flake8:
	-$(CONDA_RUN) python -m flake8 src tests

# creation of temporary __init__.py fixes duplicate module error triggered by
# multiple conftest.py in tests dir
mypy: fake-modules-create mypy-proper fake-modules-delete
mypy-proper:
	-$(CONDA_RUN) python -m mypy src tests --check-untyped-defs

fake-modules-create:
	find tests -type d -not -name '__pycache__' -exec touch {}/__init__.py \;
fake-modules-delete:
	find tests -type d -not -name '__pycache__' -exec rm {}/__init__.py \;

# Because integration tests connect to remote services, they are slow and
# unreliable, so we do not include them by default
test:
	$(CONDA_RUN) python -m pytest -l tests/unit tests/contract

# run specific tests by filtering file or test name via "-k test_filter"
# example: "make test-k-index_deletion" filters test name by "index_deletion"
test-k-%:
	$(CONDA_RUN) python -m pytest -lvvsx -k $* tests/unit tests/contract

# input: make test-vvsx
# result: conda run -p ./envs python -m pytest -lvvsx tests
test-%:
	$(CONDA_RUN) python -m pytest -l$* tests/unit tests/contract

test-failed:
	$(CONDA_RUN) python -m pytest -lv --last-failed tests

test-unit:
	$(CONDA_RUN) python -m pytest -lv tests/unit

test-integration:
	$(CONDA_RUN) python -m pytest -lv tests/integration

# usage: make test-profile-test_name
# before profiling, run `make download-index-tss-v3-en` once
test-profile-%:
	$(CONDA_RUN) python -m pytest -k $* --profile --profile-svg tests

test-docker:
	$(MAKE) docker-exists >/dev/null || $(MAKE) docker-build
	$(MAKE) docker-test-scp-exists >/dev/null || $(MAKE) docker-build-test-scp
	$(CONDA_RUN) python -m pytest -lv tests/docker

# run specific tests by filtering file or test name via "-k test_filter"
# example: "make test-k-index_deletion" filters test name by "index_deletion"
test-docker-k-%:
	$(MAKE) docker-exists >/dev/null || $(MAKE) docker-build
	$(MAKE) docker-test-scp-exists >/dev/null || $(MAKE) docker-build-test-scp
	$(CONDA_RUN) python -m pytest -lvvsx -k $* tests/docker

test-docker-%:
	$(MAKE) docker-exists >/dev/null || $(MAKE) docker-build
	$(MAKE) docker-test-scp-exists >/dev/null || $(MAKE) docker-build-test-scp
	$(CONDA_RUN) python -m pytest -l$* tests/docker

check: format lint test
	$(CONDA_RUN) python -m pytest -lv tests/integration

# install conda:
# https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links
# install mamba: conda install mamba -n base -c conda-forge
env-create: env-create-proper env-patch

env-create-39:
	mamba env create -p ./envs39 -f env.yml

env-create-proper:
	mamba env create -p ./envs -f env.yml

env-patch:
	cp -r ./envs.patch/* ./envs39

env-update:
	mamba env update -p ./envs -f environment.yml

env-update-39:
	mamba env update -p ./envs39 -f env.yml

env-remove:
	conda env remove -p ./envs

env-delete: env-remove

doc:
	$(MAKE) -C ./docs html

run-indexer:
	$(CONDA_RUN) python src/components/indexer/watcher.py --log-dir src/components/indexer/.log --storage-dir src/components/indexer/data

run-indexer-%:
	$(CONDA_RUN) python src/components/indexer/watcher.py --log-dir src/components/indexer/.log --storage-dir src/components/indexer/data --debug false --api-key $*

run-indexer-openai-%:
	$(CONDA_RUN) python src/components/indexer/watcher.py --log-dir src/components/indexer/.log --storage-dir src/components/indexer/data --debug false --provider openai --api-key $*

run-indexer-deepgram-%:
	$(CONDA_RUN) python src/components/indexer/watcher.py --log-dir src/components/indexer/.log --storage-dir src/components/indexer/data --debug false --provider deepgram --api-key $*

lab:
	$(CONDA_RUN) python -m jupyter lab --no-browser

SHELL := $(shell which bash)

# Q: why `$(CONDA_RUN) some_command` instead of just
# `conda run -p ./envs --no-capture-output some_command`
# A: code executed via conda run does not exit on interrupt
# https://github.com/conda/conda/issues/11420
# CONDA_ACTIVATE implementation was borrowed from
# https://stackoverflow.com/a/71548453/6656775
.ONESHELL:
CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate
CONDA_RUN = $(CONDA_ACTIVATE) ./envs39;
