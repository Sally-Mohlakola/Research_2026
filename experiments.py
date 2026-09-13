#!/usr/bin/env python3
"""
experiments.py -- ablation matrix for the neural diamond RDM study, driven by SLURM.

Two phases, because gathers are shared between runs and SLURM array tasks run
concurrently. Phase 1 produces every unique RDM; phase 2 trains and renders on
top of them. Splitting it this way means no two tasks ever write the same file,
so there are no locks and no races.

    python experiments.py --list                     # show the manifest
    python experiments.py --stage gather --index 0   # one gather
    python experiments.py --stage run    --index 0   # one train + its renders
    python experiments.py --collect                  # assemble results table
    python experiments.py --emit-sbatch              # write slurm/*.sbatch

Every stage is skippable: if the output already exists it is left alone, so a
failed array job can be resubmitted wholesale without redoing finished work.
Timings and metrics land in results/<name>.json, and each checkpoint gets a
commands.md recording exactly what produced it -- this project has ~100 render
directories whose provenance is already unrecoverable, and that is a mistake
worth not repeating.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable                      # whatever python SLURM gave us
CKPT = ROOT / "checkpoints"
RENDERS = ROOT / "renders"
LOGS = ROOT / "logs"
RESULTS = ROOT / "results"

# Reference render (explicit path tracing, no neural term) that every run is
# scored against. Regenerate with eval.py --no_neural if it is missing.
REFERENCE = "r18_ref"

DIAMOND = "round_diamond_gia"

# Render settings shared by every run, so images are comparable across the
# matrix. spp is deliberately modest: the matrix is wide, and the metrics that
# matter (contrast, chroma, correlation) are stable well before the image is
# visually converged.
RENDER = dict(width=512, height=512, spp=32, frames=1, max_depth=64)


# ─────────────────────────────────────────────────────────────────────────────
# The matrix
# ─────────────────────────────────────────────────────────────────────────────

def build_manifest():
    """
    Returns (gathers, runs).

    gathers: name -> gather kwargs. One entry per *unique* RDM.
    runs:    list of dicts, each naming a gather plus a training config and the
             renders to make from it.

    A gather is ~2-7 minutes at the measured 261k rays/s; a training at
    width 128 / bands 6 is tens of minutes. So the matrix is deliberately
    arranged to share gathers wherever the RDM would be identical -- the
    architecture sweep in particular is four trainings on one gather.
    """
    gathers, runs = {}, []

    def gather(name, theta, phi, k=8, batches=30, dispersion=True,
               diamond=DIAMOND, max_samples=0):
        """
        Register a gather, reusing an existing one with identical parameters.

        Several sweeps share a centre point -- the resolution sweep's 32x64
        entry, the k sweep's k=8, the architecture sweep's base, the ray-budget
        sweep's 30M and the geometry sweep's gia stone are all the same RDM.
        Without this they would be gathered five times, costing ~10 minutes and
        850 MB of duplicate histograms, and worse, each would have different
        gather noise, so a difference between two sweeps could come from the
        RDM rather than from the variable under test. Sharing the checkpoint
        makes the sweeps genuinely comparable at their shared point.
        """
        cfg = dict(theta_bins=theta, phi_bins=phi, k=k, num_batches=batches,
                   dispersion=dispersion, diamond=diamond,
                   max_samples=max_samples)
        for existing, other in gathers.items():
            if other == cfg:
                return existing
        gathers[name] = cfg
        return name

    def run(name, gather_name, width=128, bands=6, epochs_m=None,
            renders=("routeA", "routeA_off", "agg", "agg_off", "sparkle")):
        runs.append(dict(name=name, gather=gather_name, width=width,
                         bands=bands, epochs_m=epochs_m, renders=list(renders)))

    # -- A. Resolution sweep -------------------------------------------------
    # The experiment that decides whether the failure is a property of the
    # method or of running it at 1/979th the specified resolution. Soh &
    # Montazeri specify 22x90x45x90, which is --theta_bins 45 --phi_bins 90.
    for theta, phi in [(8, 16), (16, 32), (24, 48), (32, 64), (45, 90)]:
        g = gather(f"res_{theta}x{phi}", theta, phi, k=8)
        run(f"res_{theta}x{phi}", g)

    # -- B. Bounce-count boundary -------------------------------------------
    # Redone at a sane resolution: the existing k-sweep was measured at 22.5 deg
    # cells through a bands=0 network, so both confounds applied at once.
    for k in [1, 2, 4, 6, 8]:
        g = gather(f"k{k}", 32, 64, k=k)
        run(f"k{k}", g)

    # -- C. Architecture ------------------------------------------------------
    # Four trainings on ONE gather. Offline this gave 64.0 / 63.0 / 64.1 / 96.5%
    # of the RDM's saturation -- width and encoding only work together. This
    # repeats it in rendered images, which is the stronger form.
    g_arch = gather("arch_base", 32, 64, k=8)
    for w, b in [(21, 0), (128, 0), (21, 6), (128, 6)]:
        run(f"arch_w{w}_b{b}", g_arch, width=w, bands=b)

    # -- D. Gather convergence ------------------------------------------------
    # Shows the RDM is converged and the failure is not undersampling.
    for nb in [12, 30, 100]:
        g = gather(f"rays_{nb}M", 32, 64, k=8, batches=nb)
        run(f"rays_{nb}M", g)

    # -- E. Geometry robustness ----------------------------------------------
    # Everything in the project so far rests on one stone and one camera.
    for stone in ["round_diamond_gia", "round_diamond_deep",
                  "round_diamond_sharp_culet"]:
        g = gather(f"geom_{stone}", 32, 64, k=8, diamond=stone)
        run(f"geom_{stone}", g)

    # -- F. Dispersion control ------------------------------------------------
    # Without --dispersion the RDM's per-bin channel spread is exactly zero, so
    # this isolates fire from everything else.
    g = gather("nodisp", 32, 64, k=8, dispersion=False)
    run("nodisp", g)

    return gathers, runs


# Render variants. Each maps to a set of eval.py flags. `sparkle` needs a
# checkpoint gathered after var_m existed, which every gather here is.
RENDER_VARIANTS = {
    "routeA":     [],
    "routeA_off": ["--clamp_value", "0"],
    "agg":        ["--no_explicit_entry"],
    "agg_off":    ["--no_explicit_entry", "--clamp_value", "0"],
    "sparkle":    ["--no_explicit_entry", "--sparkle"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Execution
# ─────────────────────────────────────────────────────────────────────────────

def sh(cmd, log_path):
    """Run a command, tee to a log, return (ok, seconds)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n$ " + " ".join(str(c) for c in cmd) + "\n")
        f.flush()
        p = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    return p.returncode == 0, time.time() - t0


def do_gather(name, cfg):
    out = CKPT / name
    if (out / "rdm.npz").exists():
        print(f"[skip] gather {name} (rdm.npz exists)")
        return True, 0.0
    cmd = [PY, "gather_rdm.py", "--checkpoint_name", name,
           "--diamond_name", cfg["diamond"],
           "--batch_size", "1000000", "--num_batches", str(cfg["num_batches"]),
           "--theta_bins", str(cfg["theta_bins"]), "--phi_bins", str(cfg["phi_bins"]),
           "--max_depth", "64", "--k", str(cfg["k"])]
    if cfg["dispersion"]:
        cmd.append("--dispersion")
    if cfg["max_samples"]:
        cmd += ["--max_samples", str(cfg["max_samples"])]
    print(f"[gather] {name}")
    return sh(cmd, LOGS / f"gather_{name}.log")


def epochs_for(cells, batch=4096, target_steps=160_000):
    """
    Hold the number of gradient steps roughly constant across resolutions.

    Epoch count is meaningless on its own here: 4000 epochs over 8,192 bins and
    4000 over 8,019,000 differ by three orders of magnitude in optimisation
    work. Fixing total steps is what makes a resolution sweep an experiment
    about resolution rather than about training budget.
    """
    per_epoch = max(1, cells // batch)
    # The upper bound is deliberately loose. A tight cap (5000) binds at the
    # coarsest resolution, where 8,192 cells give only 2 batches per epoch: the
    # 8x16 run would then get 10,000 gradient steps against 160,000 for every
    # other point, and the resolution sweep would be measuring training budget
    # as much as resolution. Small-epoch overhead is negligible here -- an epoch
    # at 8,192 cells is one permutation and two batches.
    return max(20, min(200_000, target_steps // per_epoch))


def do_run(spec, gathers):
    name, gname = spec["name"], spec["gather"]
    gcfg = gathers[gname]
    src, dst = CKPT / gname, CKPT / name
    dst.mkdir(parents=True, exist_ok=True)

    if not (src / "rdm.npz").exists():
        print(f"[fail] gather {gname} missing for run {name}")
        return False
    # Hard-link the histogram rather than copying: at 45x90 it is 642 MB, and
    # the architecture sweep alone would otherwise duplicate it four times.
    tgt = dst / "rdm.npz"
    if not tgt.exists():
        try:
            os.link(src / "rdm.npz", tgt)
        except OSError:
            import shutil
            shutil.copy2(src / "rdm.npz", tgt)

    timing = {}
    cells = (gcfg["theta_bins"] // 2) * gcfg["phi_bins"] * \
            gcfg["theta_bins"] * gcfg["phi_bins"]
    epochs = spec["epochs_m"] or epochs_for(cells)

    if (dst / "model_m.pth").exists():
        print(f"[skip] train {name}")
    else:
        cmd = [PY, "train_models.py", "--checkpoint_name", name,
               "--diamond_name", gcfg["diamond"],
               "--width", str(spec["width"]), "--bands", str(spec["bands"]),
               "--epochs_m", str(epochs), "--epochs_t", "2000",
               "--batch_size", "4096", "--no_plot"]
        print(f"[train] {name}  ({cells} cells, {epochs} epochs)")
        ok, secs = sh(cmd, LOGS / f"train_{name}.log")
        timing["train"] = secs
        if not ok:
            print(f"[fail] train {name}")
            return False

    # A bands>0 network makes the first layer 6*(1+2*bands) wide, and
    # drjit_wrapper materialises out x in x lanes -- 83.8 GB at 128x78 for a
    # full 512x512 wavefront. Tiling is what makes it fit.
    tile = 128 if (spec["bands"] > 0 or spec["width"] > 32) else 0

    for variant in spec["renders"]:
        outdir = RENDERS / f"{name}__{variant}"
        if (outdir / "frames" / "frame_0000.exr").exists():
            print(f"[skip] render {name} {variant}")
            continue
        cmd = [PY, "eval.py", "--checkpoint_name", name,
               "--diamond_name", gcfg["diamond"],
               "--frames", str(RENDER["frames"]), "--spp", str(RENDER["spp"]),
               "--width", str(RENDER["width"]), "--height", str(RENDER["height"]),
               "--max_depth", str(RENDER["max_depth"]),
               "--output_dir", str(outdir.relative_to(ROOT))]
        if tile:
            cmd += ["--tile", str(tile)]
        cmd += RENDER_VARIANTS[variant]
        print(f"[render] {name} {variant}")
        ok, secs = sh(cmd, LOGS / f"render_{name}_{variant}.log")
        timing[f"render_{variant}"] = secs
        if not ok:
            print(f"[warn] render {name} {variant} failed; continuing")

    write_commands_md(dst, name, spec, gcfg, epochs, tile)
    score(name, spec, gcfg, timing)
    return True


def write_commands_md(dst, name, spec, gcfg, epochs, tile):
    disp = " --dispersion" if gcfg["dispersion"] else ""
    ms = f" --max_samples {gcfg['max_samples']}" if gcfg["max_samples"] else ""
    txt = f"""Checkpoint {name} -- produced by experiments.py.

GATHER
python gather_rdm.py --checkpoint_name {spec['gather']} --diamond_name {gcfg['diamond']} \\
  --batch_size 1000000 --num_batches {gcfg['num_batches']} \\
  --theta_bins {gcfg['theta_bins']} --phi_bins {gcfg['phi_bins']} \\
  --max_depth 64 --k {gcfg['k']}{disp}{ms}

TRAIN
python train_models.py --checkpoint_name {name} --diamond_name {gcfg['diamond']} \\
  --width {spec['width']} --bands {spec['bands']} --epochs_m {epochs} \\
  --epochs_t 2000 --batch_size 4096 --no_plot

RENDER (one per variant; {'tiled at ' + str(tile) if tile else 'untiled'})
"""
    for v in spec["renders"]:
        flags = " ".join(RENDER_VARIANTS[v])
        t = f" --tile {tile}" if tile else ""
        txt += (f"  {v}: python eval.py --checkpoint_name {name} "
                f"--diamond_name {gcfg['diamond']} --frames 1 --spp {RENDER['spp']} "
                f"--width {RENDER['width']} --height {RENDER['height']} "
                f"--max_depth 64{t} {flags} "
                f"--output_dir renders/{name}__{v}\n")
    txt += f"""
Bins: {gcfg['theta_bins'] // 2}x{gcfg['phi_bins']} incoming x {gcfg['theta_bins']}x{gcfg['phi_bins']} outgoing
      = {(gcfg['theta_bins'] // 2) * gcfg['phi_bins'] * gcfg['theta_bins'] * gcfg['phi_bins']} cells,
      {90 / (gcfg['theta_bins'] // 2):.2f} x {360 / gcfg['phi_bins']:.2f} deg incoming cells.
Rays: {gcfg['num_batches']}M. Architecture: width {spec['width']}, bands {spec['bands']}.

Epoch count is derived from cell count to hold gradient steps near constant
across the resolution sweep -- see epochs_for() in experiments.py. Comparing
runs at fixed epochs would confound resolution with training budget.
"""
    (dst / "commands.md").write_text(txt, encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _load(path):
    import mitsuba as mi
    import numpy as np
    return np.array(mi.Bitmap(str(path)))[..., :3]


def score(name, spec, gcfg, timing):
    """Score every render of one run against the reference, into results/."""
    import config  # noqa: F401  -- must precede mitsuba, see eval.py
    import mitsuba as mi
    import numpy as np
    if mi.variant() is None:
        mi.set_variant("scalar_rgb")
    W = np.array([0.2126, 0.7152, 0.0722])

    ref_path = RENDERS / REFERENCE / "frames" / "frame_0000.exr"
    ref = _load(ref_path) @ W if ref_path.exists() else None
    mask = ref > np.percentile(ref, 55) if ref is not None else None

    out = dict(name=name, width=spec["width"], bands=spec["bands"],
               gather=spec["gather"], timing=timing, **gcfg)
    for variant in spec["renders"]:
        p = RENDERS / f"{name}__{variant}" / "frames" / "frame_0000.exr"
        if not p.exists():
            continue
        b = _load(p)
        l = b @ W
        nz = l[l > 1e-6]
        mx, mn = b.reshape(-1, 3).max(1), b.reshape(-1, 3).min(1)
        ok = mx > 1e-6
        f = b.reshape(-1, 3)
        s = f[f.sum(1) > 1e-6].sum(0)
        c = s / s.sum()
        m = dict(
            mean=float(l.mean()),
            contrast=float(np.percentile(l, 99) / max(np.median(nz), 1e-12)),
            top1=float(np.sort(l.ravel())[-l.size // 100:].sum() / max(l.sum(), 1e-12)),
            saturation=float(((mx[ok] - mn[ok]) / mx[ok]).mean()),
            r_over_g=float(c[0] / c[1]), r_over_b=float(c[0] / c[2]),
            max=float(l.max()),
        )
        if ref is not None and l.shape == ref.shape:
            m["corr_ref"] = float(np.corrcoef(l[mask], ref[mask])[0, 1])
        out[variant] = m

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


def collect():
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(RESULTS.glob("*.json"))]
    if not rows:
        print("no results yet")
        return
    hdr = f"{'run':<26}{'w':>4}{'b':>3}{'cells':>10}  " \
          f"{'variant':<11}{'contrast':>9}{'satur.':>8}{'R/G':>7}{'corr':>8}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        cells = (r["theta_bins"] // 2) * r["phi_bins"] * r["theta_bins"] * r["phi_bins"]
        for v in RENDER_VARIANTS:
            if v not in r:
                continue
            m = r[v]
            print(f"{r['name']:<26}{r['width']:>4}{r['bands']:>3}{cells:>10}  "
                  f"{v:<11}{m['contrast']:>9.2f}{m['saturation']:>8.4f}"
                  f"{m['r_over_g']:>7.3f}{m.get('corr_ref', float('nan')):>8.4f}")
    csv = RESULTS / "summary.csv"
    import csv as _csv
    with open(csv, "w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(["run", "width", "bands", "cells", "k", "rays_M", "diamond",
                    "dispersion", "variant", "mean", "contrast", "top1",
                    "saturation", "r_over_g", "r_over_b", "max", "corr_ref"])
        for r in rows:
            cells = (r["theta_bins"] // 2) * r["phi_bins"] * r["theta_bins"] * r["phi_bins"]
            for v in RENDER_VARIANTS:
                if v not in r:
                    continue
                m = r[v]
                w.writerow([r["name"], r["width"], r["bands"], cells, r["k"],
                            r["num_batches"], r["diamond"], r["dispersion"], v,
                            m["mean"], m["contrast"], m["top1"], m["saturation"],
                            m["r_over_g"], m["r_over_b"], m["max"],
                            m.get("corr_ref", "")])
    print(f"\nwrote {csv}")


# ─────────────────────────────────────────────────────────────────────────────
# SLURM
# ─────────────────────────────────────────────────────────────────────────────

SBATCH = """#!/bin/bash
#SBATCH --job-name=rdm_{stage}
#SBATCH --array=0-{last}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={walltime}
#SBATCH --output=logs/slurm_{stage}_%A_%a.out

# Dr.Jit resolves the LLVM shared library on first JIT init and caches the
# result, so config must be imported before mitsuba -- experiments.py and the
# scripts it calls already do that. If the cluster's LLVM is elsewhere, set
# DRJIT_LIBLLVM_PATH here and config/__init__.py will respect it.
# export DRJIT_LIBLLVM_PATH=/path/to/libLLVM.so

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun python experiments.py --stage {stage} --index $SLURM_ARRAY_TASK_ID
"""


def emit_sbatch(n_gathers, n_runs):
    d = ROOT / "slurm"
    d.mkdir(exist_ok=True)
    (d / "gather.sbatch").write_text(SBATCH.format(
        stage="gather", last=n_gathers - 1, cpus=16, mem="24G",
        walltime="02:00:00"), encoding="utf-8")
    # Training dominates here, and the 45x90 gather needs headroom: 8.02M cells
    # x 3 channels is ~96 MB per histogram before the training arrays.
    (d / "run.sbatch").write_text(SBATCH.format(
        stage="run", last=n_runs - 1, cpus=16, mem="48G",
        walltime="12:00:00"), encoding="utf-8")
    (d / "submit.sh").write_text(
        "#!/bin/bash\n"
        "set -e\n"
        "JID=$(sbatch --parsable slurm/gather.sbatch)\n"
        "echo \"gathers: $JID\"\n"
        "sbatch --dependency=afterok:$JID slurm/run.sbatch\n",
        encoding="utf-8")
    print(f"wrote {d}/gather.sbatch, {d}/run.sbatch, {d}/submit.sh")
    print("submit with:  bash slurm/submit.sh")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["gather", "run"])
    ap.add_argument("--index", type=int)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--emit-sbatch", action="store_true")
    a = ap.parse_args()

    gathers, runs = build_manifest()
    gnames = list(gathers)

    if a.list:
        print(f"{len(gnames)} gathers:")
        for i, n in enumerate(gnames):
            g = gathers[n]
            cells = (g["theta_bins"] // 2) * g["phi_bins"] * g["theta_bins"] * g["phi_bins"]
            print(f"  [{i:2d}] {n:<28} {g['theta_bins']}x{g['phi_bins']} "
                  f"({cells:>9,} cells) k={g['k']} {g['num_batches']}M "
                  f"{'disp' if g['dispersion'] else 'nodisp'} {g['diamond']}")
        print(f"\n{len(runs)} runs:")
        for i, r in enumerate(runs):
            print(f"  [{i:2d}] {r['name']:<28} gather={r['gather']:<24} "
                  f"w={r['width']} b={r['bands']} renders={len(r['renders'])}")
        return

    if a.emit_sbatch:
        emit_sbatch(len(gnames), len(runs))
        return

    if a.collect:
        collect()
        return

    if a.stage is None or a.index is None:
        ap.error("need --stage and --index (or --list / --collect / --emit-sbatch)")

    LOGS.mkdir(exist_ok=True)
    if a.stage == "gather":
        n = gnames[a.index]
        ok, secs = do_gather(n, gathers[n])
        print(f"gather {n}: {'ok' if ok else 'FAILED'} in {secs:.0f}s")
        sys.exit(0 if ok else 1)
    else:
        spec = runs[a.index]
        ok = do_run(spec, gathers)
        print(f"run {spec['name']}: {'ok' if ok else 'FAILED'}")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
