#!/bin/bash
set -e
JID=$(sbatch --parsable slurm/gather.sbatch)
echo "gathers: $JID"
sbatch --dependency=afterok:$JID slurm/run.sbatch
