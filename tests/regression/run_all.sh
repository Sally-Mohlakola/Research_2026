#!/usr/bin/env bash
# Run every pipeline script, the analysis tools and one GIA render from gather to render, into $1.
# Usage, from anywhere:
#   bash tests/regression/run_all.sh tests/regression/output/new
#   python -B tests/regression/compare.py tests/regression/reference tests/regression/output/new
# Make the reference once, on code you trust:
#   bash tests/regression/run_all.sh tests/regression/reference
set -e
if [ -z "$1" ]; then echo "usage: bash tests/regression/run_all.sh <output folder>"; exit 2; fi
mkdir -p "$(dirname "$1")"
OUT="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HIST="$REPO/../checkpoint_history"
cd "$REPO"
rm -rf "$OUT"; mkdir -p "$OUT"
P="python -B"
step() { echo "--- $1"; }

step unit_tests
python -B -m unittest discover -s tests 2>&1 | tail -3

step gather_learned_interior
$P pipeline/gather_learned_interior.py --output $OUT/pool.npz --entries 256 --paths_per_entry 4 --seed 9 --workers 2 --entry_sampling camera
$P pipeline/gather_learned_interior.py --output $OUT/pool_uniform.npz --entries 64 --paths_per_entry 4 --seed 9 --no_dispersion

step gather_sharded
$P pipeline/gather_sharded.py --output_dir $OUT/shards --records 2048 --paths_per_entry 4 --entries_per_shard 128 --workers 2 --seed 5

step gather_dispersion
$P pipeline/gather_dispersion.py --output $OUT/disp_train.npz --entries 200 --workers 2 --seed 3
$P pipeline/gather_dispersion.py --output $OUT/disp_val.npz --entries 100 --workers 2 --seed 4 --wavelengths rotated --entry_sampling camera

step train_learned_interior
for h in mixture clone clone_deep; do
  $P pipeline/train_learned_interior.py --data $OUT/pool.npz --output $OUT/train_$h --head $h --epochs 2
done

step train_streaming
$P pipeline/train_streaming.py --shards $OUT/shards --output $OUT/stream --epochs 1 --validation_shards 1 --group 2 --validate_every 1 --threads 1
$P pipeline/train_streaming.py --shards $OUT/shards --output $OUT/stream_init --init $OUT/stream/model.pt --epochs 1 --validation_shards 1 --group 2 --validate_every 1 --threads 1 --head clone_deep

step train_dispersion
$P pipeline/train_dispersion.py --train $OUT/disp_train.npz --validation $OUT/disp_val.npz --output $OUT/dtrain --epochs 1 --threads 1 --batch_size 256

step render_learned_interior
M=$OUT/train_mixture/model.pt
R="$P pipeline/render_learned_interior.py --width 24 --height 24 --spp 2"
$R --model $M --output_dir $OUT/r_neural
$R --model $M --output_dir $OUT/r_analytic --mode analytic
$R --model $M --output_dir $OUT/r_oracle --oracle_facet --rfilter gaussian
$R --model $OUT/train_clone_deep/model.pt --output_dir $OUT/r_split --split_components --stratify_wavelength --deterministic_exit
$R --model $M --output_dir $OUT/r_tumble --rotation_deg 20 10 5 --azimuth 30 --rfilter tent
$R --model $M --output_dir $OUT/r_spin --camera spin
$R --model $HIST/pear_model/model.pt --output_dir $OUT/r_disp --dispersion $HIST/dispersion_pear/head/model.pt
$R --model $HIST/boundary_model_pilot/model.pt --output_dir $OUT/r_prior --rdm_prior $HIST/boundary_model_pilot/rdm_prior.npz

step render_paired
$P pipeline/render_paired.py --model $M --output_dir $OUT/paired --width 24 --height 24 --spp 2 --seeds 1 2 --workers 2
$P pipeline/render_paired.py --model $OUT/train_clone/model.pt --output_dir $OUT/paired_n --width 24 --height 24 --spp 2 --seeds 3 4 --workers 2 --modes neural --skip_evaluation --split_components

step merge_renders
$P pipeline/merge_renders.py --input_dir $OUT/paired/neural --output $OUT/merged.exr

step animate_learned_interior
$P pipeline/animate_learned_interior.py --model $M --output_dir $OUT/anim --frames 2 --width 16 --height 16 --spp 1 --workers 2 --fps 2
$P pipeline/animate_learned_interior.py --model $M --output_dir $OUT/anim_t --frames 2 --width 16 --height 16 --spp 1 --workers 2 --fps 2 --motion tumble --vary_seed --mode analytic

step retarget_checkpoint
$P pipeline/retarget_checkpoint.py --model $HIST/pear_model/model.pt --diamond_name round_diamond_gia --output $OUT/retarget/model.pt --allow_resize
$P pipeline/retarget_checkpoint.py --model $M --diamond_name round_diamond_gia --output $OUT/retarget_id/model.pt --correspondence identity

# One GIA render of the learned interior operator, from gather to render, with the whole analysis:
# a training pool and an independent test pool, the stochastic operator trained to convergence and rendered
# against the analytic reference, the deep decoder scored on the same reference pixels and compared with it,
# quality against time for both, and the paired seed summary. The runs go to figures/, checkpoints/ and
# renders/ as real runs do, then move into $OUT so the reference comparison covers them.
for parent in figures checkpoints renders; do rm -rf $parent/harness_gia_*; done

step gia_gather
$P pipeline/gather_learned_interior.py --output $OUT/gia_pool.npz --diamond_name round_diamond_gia --entries 8192 --paths_per_entry 16 --max_depth 128 --seed 801 --workers 8 --entry_sampling camera --azimuth_range 0 360
$P pipeline/gather_learned_interior.py --output $OUT/gia_test.npz --diamond_name round_diamond_gia --entries 2048 --paths_per_entry 16 --max_depth 128 --seed 802 --workers 8 --entry_sampling camera --azimuth_range 0 360

step gia_stochastic_train_render_evaluate
G="--pool $OUT/gia_pool.npz --test_pool $OUT/gia_test.npz --no_index --image_width 128 --image_height 128 --spp 8 --seeds 101 102 103 104 --workers 4 --benchmark_repeats 10"
$P pipeline/run_experiment.py --name harness_gia_stochastic --head mixture --note "harness: GIA, stochastic decoder" $G

step gia_deep_decoder_against_the_same_reference
$P pipeline/run_experiment.py --name harness_gia_deep --head clone_deep --reference_run harness_gia_stochastic --note "harness: GIA, deep decoder" $G

step gia_analysis
$P analysis/compare_runs.py --runs harness_gia_deep harness_gia_stochastic
$P analysis/quality_vs_time.py --run harness_gia_stochastic
$P analysis/aggregate_runs.py --name harness_gia_deep_group --runs harness_gia_deep --against harness_gia_stochastic
$P figures/plot_run.py --run figures/harness_gia_stochastic --renders renders/harness_gia_stochastic
$P pipeline/run_paper_batch.py --dry_run > $OUT/paper_batch_plan.txt

for parent in figures checkpoints renders; do
  mkdir -p $OUT/$parent
  for run in $parent/harness_gia_*; do mv "$run" $OUT/$parent/; done
done

echo ALL DONE
