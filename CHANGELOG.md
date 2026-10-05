# Changelog

Notable changes to `emu_pk`. Format follows [Keep a Changelog](https://keepachangelog.com/1.1.0/);
versioning is [semantic](https://semver.org/spec/v2.0.0.html), and from 1.0.0
the public API is what `emu_pk.__all__` and each module's `__all__` declare.

## [2.1.0]

A training truth that is converged, not default, and a grid that reaches
300 h/Mpc. The network is unchanged in form; what it is trained on changes, and
the weights are scored against a reference neither version was trained on --
CAMB at its converged rung, times the same reionization heating
(`emu_pk/data/validation_reference.json`). Total error in $P_m$ at $z = 0$,
32 held-out cosmologies:

| against the converged reference | median | 90th | max |
|---|---|---|---|
| 2.0.1, $k \in [10^{-3}, 10]$ | 0.351 % | 0.580 % | 1.358 % |
| **2.1.0**, $k \in [10^{-3}, 10]$ | **0.063 %** | 0.130 % | 0.190 % |
| 2.0.1, tail to its 200 h/Mpc | 0.494 % | 0.565 % | 0.629 % |
| **2.1.0**, tail to 300 h/Mpc | **0.044 %** | 0.069 % | 0.086 % |

Against its own training truth, which is what `validation.json` records, 2.1.0
scores 0.064 % in the median; 2.0.1 scored 0.066 % against *its* truth, which
was itself 0.41 % from converged. The mass-ratio derivatives improve from 5 % to
4 %, and $\partial\ln P/\partial z$ is scored again, 16 of 16 points.

### Why

`ggah_mod_benchmark`'s precision scan (commit 516eb0a) put the 2.0 training
set -- CLASS 3.3.4 at its defaults -- 0.41 % from CLASS's own converged answer
in the median over this box and 0.81 % at worst, smooth in the parameters. The
network learnt that error faithfully, and its validation, against the same
default CLASS, could not see it: 6 to 13 times the 0.064 % it reported. No
affordable CLASS setting fixes the heavy-neutrino end (the ncdm fluid
approximation dominates above 0.3 eV).

### Changed

- **The generator solves with CAMB** (`generate.SOLVER = "camb"`) at
  `cosmo.CAMB_PRECISION` -- `lAccuracyBoost 3`, `AccuracyBoost 2`, no late
  radiation truncation, no Cls -- which is `ggah_mod` 0.9.8's: 0.039 % from
  CAMB's converged reference and 0.12 % from CLASS's, the floor between the two
  codes. `"class"` keeps the 2.0 path.
- **Reionization heating is kept.** CAMB's linear P(k) does not heat the
  baryons at reionization and CLASS's does (3 % at k = 200 h/Mpc at the
  fiducial, 13 % at omega_cdm = 0.05). Each CAMB spectrum is multiplied by
  CLASS's ratio P(reio)/P(no reio) from a CLASS pair at the same point, at
  CLASS's own defaults with the ratio taken on CLASS's own k nodes: within
  6e-5 of a pair tightened everywhere, 1.3e-4 at omega_cdm = 0.05 above
  200 h/Mpc. A point CLASS refuses is retried once at a thousandfold tighter
  tolerance. The history is CLASS's default, z_reio = 7.6711 everywhere.
- Below CAMB's first transfer mode (curved corners of the box, k < 1.5e-4 and
  3.7e-4 h/Mpc) the spectrum continues with the shape of the heated CLASS
  spectrum, pinned to CAMB's.
- **The grid reaches 300 h/Mpc**, 411 nodes at 2.0's density, where it stopped
  at 200 with 400. `ggah_mod`'s Boltzmann backends tabulate to 300.
- **`validate` scores against the truth the network learnt**, read from the
  weights (`--truth training`, the default), and names it in every table;
  `--truth reference` scores against the converged rung. The small-scale tail,
  10 h/Mpc to the network's own k_max, is scored as its own band.
- `[gen]` installs CAMB beside `classy`. The shipped weights were trained and
  validated on CAMB 1.6.6; CAMB 2.0.4 differs from it by up to 4.4e-4 in ln P
  near k = 0.1 h/Mpc, a visible fraction of the network's own error.

### Added

- `cosmo.camb_params`: `ggah_mod` 0.9.8's `camb_input` for the box, restated
  (equal to 1e-16 in every field; `tests/test_camb_path.py`). The neutrinos go in
  as each state's exact density, never as mass shares, which CAMB reads as
  density shares.
- `cosmo.nu_energy_factor`, and the 0.9.8 neutrino constants it needs.
- `heating`: the CLASS pair.
- Shards, assembled datasets and weights record which solver, precision and
  heating wrote them; `assemble` refuses a directory holding two.
- `emu_pk/data/validation_reference.json`: the shipped weights against the
  converged reference.
- Validation records carry `weights_sha256`, the checksum of the file scored,
  and a test ties both shipped records to the shipped weights by it.
- `validate --cache`: truth solves are kept between runs and shared between
  machines, and `oarsub/prefill_reference.py` fills the cache as parallel short
  jobs.
- `train --host-targets` keeps the training targets in host memory and sends
  each batch to the device, for a card smaller than the training set: the 150k
  set's targets are 13.5 GiB. Same batches, same order, same weights.
- The cluster campaign trains on a GPU (`submit_campaign.sh train-gpu`, Bigfoot)
  and scores on Dahu; generation runs many shards inside one job.
- The generation environment is checked for CAMB >= 1.6: `CAMB_PRECISION` was
  measured on 1.6.6, and the campaign's default environment carries 1.4.0.

### Fixed

- **CAMB was asked for the same redshift twice and refused.** `validate`'s
  $\partial\ln P/\partial z$ stencils need z = 0 and 0.05 at both step sizes,
  and a repeated redshift is a zero-length step for CAMB's integrator (DVERK
  error): every one of the 16 points was skipped, and the redshift derivative
  went unscored. `generate.solve_camb` now asks for each redshift once.
- **A truth cache filled on one machine missed on every other.** Its keys
  hashed the scoring wavenumbers exactly, and numpy's `exp` rounds differently
  with and without AVX-512: 21 of 300 differed in the last bit between Dahu and
  a laptop on the same numpy. Keys now hash rounded values.
- **GPU training in TF32.** On an A100, JAX multiplies float32 matrices in TF32
  unless told otherwise; the same data, seed and schedule stopped at a val loss
  of 5.9e-7 against the CPU's 3.7e-7, its worst-case error 1.6x higher.
  `run_train.sh` sets `JAX_DEFAULT_MATMUL_PRECISION=highest` on a GPU, and the
  shipped network was trained at full float32.

## [2.0.1]

### Changed

- `cosmo.NU_DENOM_EV` is `93.14338613172058`, the float `ggah_mod` 0.9.8 derives
  for the rest mass of three states at CLASS's default `T_ncdm = 0.71611`, where
  it was the rounded `93.14`. It reaches `omega_nu` and `f_nu`, which the
  correction tables and the validation read; the network and its weights are
  unchanged, so every prediction is bit for bit what 2.0.0 returns.

## [2.0.0]

Three separate neutrino masses and spatial curvature, on a retrained network.
Eleven parameters against eight, and **more accurate than 1.0.0 despite the
wider box** -- 0.064 % median shape error against 0.111 %.

### Added

- **Three neutrino masses.** `nu_r1`/`nu_r2` divide `sum_mnu` over three
  separate CLASS species, `m_i = r_i * sum_mnu`, on the ordered simplex.
  `sum_mnu` keeps its index and its meaning; what is new is how it is divided.
  Measured: the degenerate approximation is wrong by 0.32 % in P(k) at
  `sum_mnu = 0.10` eV in an inverted ordering, and by under 0.012 % above
  0.25 eV.
- **Spatial curvature.** `Omega_k` is the ninth parameter of the box, spanning
  `[-0.15, +0.15]`, positive open. It is appended to `box.PARAMS` rather than
  inserted, so every existing `PARAMS.index` keeps its value. The bound is
  measured: CLASS solves the whole box at `|Omega_k| <= 0.15`, and on the closed
  side it begins refusing near `-0.275` at the low-density corner.
- `validate.K_LOWK` scores `k` in `[1e-4, 1e-3]` separately. The curvature scale
  is inside the `k` grid, and that decade was previously generated and never
  scored.
- `validate.flat_slice_error` and `--flat-only` score the `Omega_k = 0` slice,
  which is what says whether the ninth parameter cost the other eight anything.
- `where_in_box` reports the closure `Omega_de`, and summaries carry
  `negative_de` and `curved` strata beside the existing quintessence corner.
- `box.sample(pin=...)` and `generate --pin` hold a design column fixed, which
  is how the flat control is built.
- `generate.class_params_for` maps a design row onto CLASS's keywords in one
  place, replacing three hand-written parameter lists.
- Shards stamp `box.PARAMS`, and `assemble` refuses a shard written against a
  different box — by name, so a permuted column is caught too.
- `validate.interpolation_floor` reports what the shape metric's own
  interpolation costs, the way `derivative_error` has always reported its
  finite-difference floor.
- The cluster campaign runs three design arms (`EMU_ARM=c|f|d`) into separate
  directories, and `validate --pin-score name=value` scores a control arm on
  the slice it was trained for.

### Changed

- **Breaking: `theta` is eleven elements.** Shorter vectors are refused.
- **Breaking: 1.0.0 weights will not load.** A checkpoint that has no input for
  a parameter this box samples is refused rather than silently ignoring it.
  `allow_narrow_box=True` is the deliberate escape hatch.
- `cosmo.class_params` takes `Omega_k` instead of hard-coding zero. The default
  is still `0.0`, so every existing caller produces a byte-identical dict —
  which is why the correction table did not have to be rebuilt.

### Fixed

- **A short `theta` returned a spectrum instead of raising.** JAX clamps an
  out-of-range gather, and `_validate` zipped `box.PARAMS` against `params`,
  where `zip` truncates. Measured against the shipped weights, a seven-long
  vector where eight were wanted moved `P(0.05)` by 10.3 % with nothing raised.
  Unreachable before the box grew; on every existing caller after.
- The derivatives figure zipped `box.PARAMS` against a fixed 2×4 grid of axes,
  so a ninth parameter would not have been drawn and nothing would have said so.
- `pull_results.sh` pulled a hardcoded `emu_pk_mlp.npz` — the 1.0.0 file — and
  reported success; `campaign_status.sh` audited only one design arm.
- `submit_campaign.sh` now refuses more than 94 array elements, with the value
  to use, rather than handing the rejection to the scheduler.
- **The reported accuracy was measuring the ruler.** `_interp_lnk` used
  `jnp.interp`, and linear interpolation from the 400-node grid onto the 300
  scoring points accounted for 1.01x the 1.0.0 median and 1.04x its p90 -- the
  network's own error was below what the metric could resolve. It is a fixed
  four-point cubic now, chosen over the monotone cubic in `interp.py` because
  that limiter branches on the data and the data here is network output. The
  floor falls from 0.1124 % to 0.0199 %.
- **The trainer's defaults did not build the model that ships.** `train()`
  defaulted to a PCA head on plain `z` while the shipped checkpoint declared
  `direct` and `log10_1pz`, so every cluster run trained the wrong
  configuration -- the worst of the four 1.0.0 ablations. `--direct` is now
  `--no-direct`, and tests read the shipped file and assert the defaults agree.
- **The validation split leaked.** Rows are `(cosmology, redshift)` pairs, so a
  random row split put nearly every cosmology on both sides and `val_loss`
  measured z-interpolation rather than generalisation. Whole cosmologies are
  held out now.
- **The degenerate neutrino point was outside the box in single precision.**
  `nu_r1`'s upper bound is 1/3, which follows from the ordering constraint
  rather than being chosen, and `np.float32(1/3)` lands 9.9e-9 above it -- so
  `theta` built with `jnp.array` was refused at the one point every result
  published before this release sits at. `inside` now carries two float32
  epsilons of the axis's width, which is below any width at which the fit
  changes and above the largest rounding the dtype can produce.
- **A checkpoint killed mid-write killed the run.** `_save` wrote in place, so a
  preemption during a write left a truncated file at the path the restart
  resumes from. Writes are atomic and unreadable checkpoints are skipped rather
  than fatal.

### Unchanged

- `class_pk_ratio.npz`. The correction grid is a fiducial sweep over
  `(sum_mnu, w0, wa)`, is deliberately not curved, and does not resolve the
  mass splitting -- it is indexed on `f_nu`, which depends only on the sum.

## [1.0.0]

First public release.

[Unreleased]: https://github.com/JohanComparat/emu_pk/compare/v2.1.0...HEAD
[2.1.0]: https://github.com/JohanComparat/emu_pk/compare/v2.0.1...v2.1.0
[2.0.1]: https://github.com/JohanComparat/emu_pk/compare/v2.0.0...v2.0.1
[2.0.0]: https://github.com/JohanComparat/emu_pk/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/JohanComparat/emu_pk/releases/tag/v1.0.0
