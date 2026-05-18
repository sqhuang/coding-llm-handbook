#  capstone_runtime — convenience targets
#  All targets are thin wrappers; the real logic is in steps/*.py and tools/*.

PY        ?= python3
TRACK     := $(PY) tools/track.py
TRACKER   := tracker.json

.PHONY: help setup preflight test board budget export \
        step-01 step-02 step-03 step-04 step-05 step-06 step-07 step-08 step-09 step-10 \
        step-11 step-12 step-13 step-14 step-15 step-16 step-17 step-18 step-19 \
        clean

help:
	@echo "capstone_runtime — common targets"
	@echo ""
	@echo "  make setup        install python deps (pip install -r requirements.txt)"
	@echo "  make preflight    HW/SW sanity check; safe to run anywhere"
	@echo "  make test         pytest tests/  (Mac-friendly, no GPU needed)"
	@echo "  make board        kanban board"
	@echo "  make budget       budget burn vs plan"
	@echo "  make export       write tracker_view.md"
	@echo ""
	@echo "  make step-NN      run step NN (see steps/NN_*.py for what it does)"
	@echo "                    NOTE: most step scripts require a real H100 box,"
	@echo "                    cluster, and HF_TOKEN / GH_TOKEN env vars."
	@echo ""
	@echo "  make clean        rm logs/* and tracker_view.md"

setup:
	$(PY) -m pip install -U pip
	$(PY) -m pip install -r requirements.txt

preflight:
	$(PY) tools/preflight.py

test:
	$(PY) -m pytest tests/ -q

board:
	$(TRACK) board

budget:
	$(TRACK) budget

export:
	$(TRACK) export-md

# Step runners — every step does the same dance:
#   1) track start
#   2) python steps/NN_*.py   (or bash steps/NN_*.sh)
#   3) on success, you decide and call `make done-NN HOURS=.. COST=..`
# We deliberately don't auto-mark done — only you know actual hours / cost.

step-01:; $(TRACK) start capstone-01-baseline       && $(PY)  steps/01_baseline.py       2>&1 | tee -a logs/01_baseline.log
step-02:; $(TRACK) start capstone-02-internal-bench && $(PY)  steps/02_internal_bench.py 2>&1 | tee -a logs/02_internal_bench.log
step-03:; $(TRACK) start capstone-03-data           && $(PY)  steps/03_data.py           2>&1 | tee -a logs/03_data.log
step-04:; $(TRACK) start capstone-04-decontam       && $(PY)  steps/04_decontam.py       2>&1 | tee -a logs/04_decontam.log
step-05:; $(TRACK) start capstone-05-arch-probe     && $(PY)  steps/05_arch_probe.py     2>&1 | tee -a logs/05_arch_probe.log
step-06:; $(TRACK) start capstone-06-midtraining    && bash   steps/06_midtraining.sh    2>&1 | tee -a logs/06_midtraining.log
step-07:; $(TRACK) start capstone-07-longctx        && $(PY)  steps/07_longctx.py        2>&1 | tee -a logs/07_longctx.log
step-08:; $(TRACK) start capstone-08-sft-data       && $(PY)  steps/08_sft_data.py       2>&1 | tee -a logs/08_sft_data.log
step-09:; $(TRACK) start capstone-09-sft-train      && bash   steps/09_sft_train.sh      2>&1 | tee -a logs/09_sft_train.log
step-10:; $(TRACK) start capstone-10-agent-traj     && $(PY)  steps/10_agent_traj.py     2>&1 | tee -a logs/10_agent_traj.log
step-11:; $(TRACK) start capstone-11-rl-env         && $(PY)  steps/11_rl_env.py         2>&1 | tee -a logs/11_rl_env.log
step-12:; $(TRACK) start capstone-12-rl-train       && bash   steps/12_rl_train.sh       2>&1 | tee -a logs/12_rl_train.log
step-13:; $(TRACK) start capstone-13-pub-eval       && $(PY)  steps/13_pub_eval.py       2>&1 | tee -a logs/13_pub_eval.log
step-14:; $(TRACK) start capstone-14-internal-eval  && $(PY)  steps/14_internal_eval.py  2>&1 | tee -a logs/14_internal_eval.log
step-15:; $(TRACK) start capstone-15-deploy         && bash   steps/15_deploy.sh         2>&1 | tee -a logs/15_deploy.log
step-16:; $(TRACK) start capstone-16-deploy-opt     && $(PY)  steps/16_deploy_opt.py     2>&1 | tee -a logs/16_deploy_opt.log
step-17:; $(TRACK) start capstone-17-rag            && $(PY)  steps/17_rag.py            2>&1 | tee -a logs/17_rag.log
step-18:; $(TRACK) start capstone-18-agent-demo     && $(PY)  steps/18_agent_demo.py     2>&1 | tee -a logs/18_agent_demo.log
step-19:; $(TRACK) start capstone-19-retro          && $(PY)  steps/19_retro.py          2>&1 | tee -a logs/19_retro.log

clean:
	rm -f logs/*.log tracker_view.md
