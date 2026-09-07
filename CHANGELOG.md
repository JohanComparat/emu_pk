# Changelog

Notable changes to `emu_pk`. Format follows [Keep a Changelog](https://keepachangelog.com/1.1.0/);
versioning is [semantic](https://semver.org/spec/v2.0.0.html), and from 1.0.0
the public API is what `emu_pk.__all__` and each module's `__all__` declare.

## [Unreleased]

### Added

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
- The cluster campaign runs two design arms (`EMU_ARM=c|f`) into separate
  directories.

### Changed

- **Breaking: `theta` is nine elements.** Eight-element vectors are refused.
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

### Unchanged

- `class_pk_ratio.npz`. The correction grid is a fiducial sweep over
  `(sum_mnu, w0, wa)` and is deliberately not curved.

## [1.0.0]

First public release.

[Unreleased]: https://github.com/JohanComparat/emu_pk/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/JohanComparat/emu_pk/releases/tag/v1.0.0
