SHELL := /bin/bash
.DEFAULT_GOAL := help

.PHONY: help data dataprep modelprep train report all

help:

	@echo "  make all        - run all 5 scripts via run_all.sh"


all:
	bash a_scripts/run_all.sh