# OAR / GRICAD submission scripts

Job scripts for the GRICAD clusters (OAR resource manager). Adapted from
`hod_mod/oarsub/`.

Docs: <https://gricad-doc.univ-grenoble-alpes.fr/hpc/joblaunch/job_management/>

## First: your site configuration

Nothing in this directory names an allocation, a login or a home directory.
Those live in `oarsub/site.sh`, which is git-ignored:

```bash
cp oarsub/site.sh.example oarsub/site.sh
$EDITOR oarsub/site.sh          # EMU_PK_PROJECT, EMU_PK_SSH_HOST, EMU_PK_WORK
```

`_campaign_env.sh` sources it, and every other script sources that. Without
`EMU_PK_PROJECT` the submitter stops with an instruction rather than guessing an
allocation. `EMU_PK_SSH_HOST` is needed only by the two scripts that run on your
own machine (`rsync_to_dahu.sh`, `pull_results.sh`) and is best set to a
`~/.ssh/config` Host alias, so that no login name is written down at all.

## The campaign, in order

| Gate | Family | Sizing | What it produces |
| --- | --- | --- | --- |
| 1 | `calibrate` | devel, 2 cores, 20 min | seconds/solve on a Dahu node |
| 2 | `ratio` | array 12, 2 cores, 2 h | the correction table's 300 CLASS solves |
| 3 | `emu` | array 94, 2 cores, 6 h, besteffort | the training set |
| 4 | `train` | 32 cores, 24 h, besteffort | the network weights |
| 4' | `train-gpu` | 1 GPU, 12 h, besteffort | the same, on Bigfoot |

```bash
./oarsub/submit_campaign.sh calibrate      # measure first
./oarsub/submit_campaign.sh ratio
./oarsub/submit_campaign.sh emu --devel    # one shard, smoke
./oarsub/submit_campaign.sh emu
./oarsub/campaign_status.sh                # which shards landed
./oarsub/submit_campaign.sh train
```

**Gate 1 is not a formality.** The shard count and walltimes are computed from a
measured rate. Sizing 94 array elements from a guess is how a run either
wastes an allocation or dies on walltime at 90 %.

## Which machine

| Cluster | Hardware | Use for |
| --- | --- | --- |
| **Dahu** | CPU nodes (~32 cores), OmniPath | **all of this** |
| Bigfoot | V100 / A100 | GPU work; needs `oarsub -T` + `gridtoken -i 9` |
| Luke | heterogeneous | specialised |

Generation is CLASS, which is CPU and embarrassingly parallel across shards.
Training is a 4×512 MLP over ~64 PCA components — small enough that a 32-core
Dahu node fits it in hours, so both run on Dahu. `run_train.sh` is identical on
Bigfoot; only the resource line differs. See below.

## Training arms

The target, the loss and the schedule are three separable choices, so a
comparison between them is several jobs of the same length and each has to be
attributable. `run_train.sh` takes a **tag** and passes flags through; the tag
names the weights so arms do not overwrite each other, and each arm can still
resume *itself* on a besteffort queue.

```bash
EPOCHS=240                                          ./oarsub/submit_campaign.sh train
EPOCHS=240 TAG=c2nosched  TRAIN_FLAGS=--no-schedule ./oarsub/submit_campaign.sh train
EPOCHS=240 TAG=c2noreduce TRAIN_FLAGS=--no-reduced  ./oarsub/submit_campaign.sh train
EPOCHS=240 TAG=c2noweight TRAIN_FLAGS=--no-weighted ./oarsub/submit_campaign.sh train
```

Tags name the weights: `emu_pk_mlp_<tag>.npz`. The default is `c2`, and
**`base` is the only tag that takes the unsuffixed `emu_pk_mlp.npz`**. Pass it
deliberately or not at all.

Each arm writes `emu_pk_mlp_<tag>.npz` **and** `emu_pk_mlp_<tag>.validation.json`
beside it — `run_train.sh` scores the weights it just trained, while the
environment that trained them is still loaded. Nothing else produces those
numbers, so an arm that skips it has no record of what it built.

**Read the loss curve before launching more arms.** It is in the OAR log, not
in the repository. If train sits far below val the fit is data-limited and
regenerating the redshift grid is the right next gate; if the two are still
descending together it is capacity- and schedule-limited and width is the next
thing to try. That is one `grep` that redirects a whole sweep.

```bash
grep '^  epoch' oarsub/logs/<jobid>.emupk_train.out | tail -40
```

### Whether this belongs on Bigfoot

`train-gpu` runs the identical script with a GPU resource line. It is worth the
token dance (`gridtoken -i 9`, then resubmit) when the work is several runs
rather than one. Two things to check on the first GPU job, because both fail
quietly rather than loudly:

* the job logs `jax.devices()` on its first line. If that says CPU on a GPU
  node, `JAX_PLATFORMS` is pinned — `campaign_threads` only pins it where there
  is no `nvidia-smi`, so something else set it.
* the resource line follows gricad-doc but has not been exercised from this
  repository. Smoke it with `--devel` before committing an ablation to it.

## Building the environment, on each cluster separately

**Dahu and Bigfoot do not share a home directory.** `~/miniforge3` on Dahu is
not on Bigfoot and never will be, so the environment is built twice, by two
different recipes, because the two machines offer different things. `/bettik`
*is* shared, which is why the training set does not have to be.

`campaign_activate_env` picks the activator and the environment name from the
cluster it finds itself on, and checks the dependencies **that job** needs:

```bash
campaign_activate_env          # gen/validate: numpy, jax, classy
campaign_activate_env train    # train:        numpy, jax, optax
```

The scope matters. `classy` is not on Bigfoot and is not going to be (below),
so checking for it unconditionally would abort every GPU training job before it
started, over a module it was never going to import. It still fails
immediately when something genuinely required is missing, rather than letting
every array element fail identically twenty minutes later on an `ImportError`
nobody reads. `EMU_PK_ENV` overrides the name.

**Do not use `/applis/cluster_name` to tell the clusters apart** — it reads
`luke` on both Dahu and Bigfoot, which is worse than no file at all.
`campaign_cluster` matches the hostname instead, on a prefix, because a compute
node is `dahu103` or `bigfoot7` rather than `dahu`.

### Dahu — CPU, conda in `$HOME`

Dahu has a personal miniforge at `~/miniforge3`, and `_campaign_env.sh` expects
`mamba` there (override with `MAMBA_EXE` / `MAMBA_ROOT_PREFIX`).

```bash
# on dahu.ciment
~/miniforge3/bin/mamba create -y -n emu_pk python=3.11
~/miniforge3/bin/mamba activate emu_pk
pip install numpy "jax[cpu]" optax classy
```

Verified in the `hod_mod` environment, which is the default here: numpy 2.4.6,
jax 0.10.2, optax 0.2.8, classy v3.3.4. The full test suite passes against
it.

### Bigfoot — GPU, conda from `/applis`

Bigfoot has **no** `conda`, `mamba`, `nix` or `guix` on `PATH`, and no
`~/miniforge3`. The site installation is sourced instead:

```bash
# on bigfoot.ciment
source /applis/environments/conda.sh          # site miniconda, envs -> ~/.conda/envs
conda create -y -n emu_pk python=3.11
conda activate emu_pk
pip install numpy "jax[cuda12]" optax     # deliberately no classy -- see below
```

Verified: numpy 2.4.6, jax 0.10.2 with `jax-cuda12-plugin` 0.10.2 and
`nvidia-cudnn-cu12` 9.24. Four things about it:

* **The pip wheels carry their own CUDA.** `/applis/environments/cuda_env.sh`
  offers toolkits from 10.2 to 12.6 (`source cuda_env.sh bigfoot 12.6`), and the
  site cuDNN for CUDA 12 is 8.9 — older than the 9.x current JAX wants. Letting
  pip pull `nvidia-cudnn-cu12` sidesteps the mismatch entirely and is why no
  `cuda_env.sh` line appears above. Only the *driver* has to come from the node.
* **The login node has no GPU.** `nvidia-smi` is absent there and present on the
  compute nodes, which is exactly the condition `campaign_threads` tests before
  pinning `JAX_PLATFORMS=cpu` — so an interactive check on the frontend will say
  CPU and prove nothing.
* **The job says which device it got.** `run_train.sh` prints `jax.devices()` on
  its first line, because a GPU job that quietly ran on the CPU is otherwise
  indistinguishable from a slow one. On the frontend it prints
  `[CpuDevice(id=0)]` after a `cuInit(0) failed: CUDA error 303`, and that is
  the correct answer there, not a broken install.
* **`classy` does not build on the Bigfoot frontend, and this environment does
  without it.** CLASS's `setup.py` runs `make` with a bare `-j`, so it launches
  as many `g++ -O3` processes as there are translation units; CLASS's are large,
  the frontend caps what one user may hold, and `cc1plus` is killed:

  ```
  g++: fatal error: Killed signal terminated program cc1plus
  make: *** [Makefile:110: harmonic.opp] Error 1
  ```

  `MAKEFLAGS=-j1` does not fix it — an explicit `-j` on the command line wins
  over the environment. Building on a compute node instead would work, but
  **training does not need CLASS at all**, so the Bigfoot environment is
  deliberately train-only. `run_train.sh` detects this, skips the post-training
  validation, and prints the exact command to run on Dahu instead — where the
  weights are anyway, since `/bettik` is shared and `$HOME` is not.

### Is Bigfoot worth it?

Measured, not assumed: **37 s/epoch on 32 Dahu cores**, so 240 epochs is 2.5 h
and a four-arm ablation is about ten. That fits the CPU queue. The GPU path is
documented for wider networks, not because throughput requires it at this size.
Smoke it with `--devel` before trusting the resource line.

## OAR dialect

* **`--name`, not `-n`.** This build rejects the short form.
* **The environment is not propagated to the node.** `VTAG=x oarsub -S ./s.sh`
  silently runs the default. Pass job settings as *arguments*.
* **`OAR_RES_NB_CORES` does not exist here.** `${OAR_RES_NB_CORES:-8}` therefore
  always falls through to 8 and a 16-core job threads to half its allocation.
  `campaign_ncores` in `_campaign_env.sh` counts `$OAR_NODEFILE` lines instead.
* **`$OAR_ARRAY_INDEX` is 1-based**; the generator's shards are 0-based.
  `run_generate.sh` decrements, in one place.
* **Nodes are heterogeneous.** Wall-clock is not comparable between jobs unless
  the CPU model is pinned: `-l /cpumodel=1/nodes=1/core=2,...`. Pin it when the
  timing *is* the measurement (`calibrate`); leave it free when throughput is
  (`emu`), or the job waits for a specific machine for no reason.
* **Max Dahu walltime is 48 h.** Everything here fits well inside.
* `-t devel` is capped at 30 minutes and is the right way to prove the
  environment before committing the whole array to it.

## `/bettik` is slow, and that is a design constraint

Measured, not assumed: one 3.0 MB compressed shard takes **3.8 s** to open and
decompress from `/bettik` on this account. That is 0.8 MB/s, and it is latency
rather than CPU -- so threads recover it and processes are not needed.

Two things in this repository exist because of that number:

* **The 3000 input shards are read in parallel** (`assemble.build_training_set`,
  16 threads). Serially the assembly takes just over three hours and reports
  nothing while it does, so there is no way to tell slow from hung. In parallel
  it takes **411 s**.
* **The assembled training set is written in 32 parts**, and
  `assemble.load_training_set` reads them in parallel. As one 9.7 GB file it
  takes over an hour to `np.load`, *once per job* -- and a besteffort job
  restarts, so that hour is paid again on every preemption.

On this filesystem, any step that reads or writes more than a gigabyte needs to
say how far it has got and needs to be parallel, or it is indistinguishable
from a hang.

## Why `besteffort` is safe for `emu` but not for `ratio`

`besteffort` jobs run in otherwise-idle capacity and are killed when a paying
job wants the node. That is only acceptable when a kill is cheap, and here it
is: **every shard skips if its output already exists**, so a killed array
element re-runs and costs only the work it had not finished. Combined with
`idempotent`, OAR restarts them automatically.

`ratio` is 300 solves — small enough that queueing for an ordinary slot lands
sooner than being repeatedly restarted out of a cheap one.

## Where things live

Shards go to `/bettik/PROJECTS/$EMU_PK_PROJECT/$USER/emu_pk` — the parallel
BeeGFS scratch, which is provisioned per project: a user cannot create
`/bettik/$USER`, only write under their project's tree. `$HOME` is neither
large enough nor meant for tens of GB. Override the whole path with
`EMU_PK_WORK` in `site.sh`. Only the assembled products come back
(`pull_results.sh`), because they are a few MB and the shards are reproducible
from a seed and an index.

Set `EMU_PK_ENV` if your conda environment is not `hod_mod`; it must carry
`classy`, `jax`, `numpy` and `optax`. `campaign_activate_env` checks and fails
immediately rather than letting every array element fail identically twenty
minutes later on an `ImportError` nobody reads.

## Monitoring

```bash
oarstat -u $USER
oarstat -fj <jobid>
oardel <jobid>
tail -f oarsub/logs/<jobid>.emupk_gen.out
./oarsub/campaign_status.sh          # names missing shards rather than counting files
```
