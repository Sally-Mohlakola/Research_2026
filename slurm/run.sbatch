#!/bin/bash
#SBATCH --job-name=rdm_run
#SBATCH --array=0-20
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm_run_%A_%a.out

# Dr.Jit resolves the LLVM shared library on first JIT init and caches the
# result, so config must be imported before mitsuba -- experiments.py and the
# scripts it calls already do that. If the cluster's LLVM is elsewhere, set
# DRJIT_LIBLLVM_PATH here and config/__init__.py will respect it.
# export DRJIT_LIBLLVM_PATH=/path/to/libLLVM.so

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun python experiments.py --stage run --index $SLURM_ARRAY_TASK_ID
