SHELL := /bin/bash
.DEFAULT_GOAL := help

.PHONY: help data dataprep modelprep train report all

help:
	@echo "Targets:"
	@echo "  make data       - run a_scripts/01_make_data.sh"
	@echo "  make dataprep   - run a_scripts/02_dataprep.sh"
	@echo "  make modelprep  - run a_scripts/03_modelprep.sh"
	@echo "  make train      - run a_scripts/04_train_hybrid.sh"
	@echo "  make report     - run a_scripts/05_eval_report.sh"
	@echo "  make all        - run all 5 scripts via run_all.sh"

data:
	bash a_scripts/01_make_data.sh

dataprep:
	bash a_scripts/02_dataprep.sh

modelprep:
	bash a_scripts/03_modelprep.sh

train:
	bash a_scripts/04_train_hybrid.sh

report:
	bash a_scripts/05_eval_report.sh

all:
	bash a_scripts/run_all.sh