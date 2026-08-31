# API reference

The supported surface is what `emu_pk.__all__` and each module's own `__all__`
list. Anything else is private and may change without a major version.

```{toctree}
:maxdepth: 1

model
box
grid
cosmo
ratio
interp
generate
assemble
train
validate
```

## At a glance

| | |
|---|---|
| {py:class}`emu_pk.PkEmulator <emu_pk.model.PkEmulator>` | evaluate a trained network |
| {py:func}`emu_pk.primordial_ln_pk <emu_pk.model.primordial_ln_pk>` | the closed-form part of $\ln P$ |
| {py:mod}`emu_pk.box` | the training hypercube and the guard |
| {py:mod}`emu_pk.grid` | the wavenumber and redshift grids |
| {py:mod}`emu_pk.cosmo` | density conventions and the CLASS input dict |
| {py:mod}`emu_pk.ratio` | the massive-ν and CPL correction table |
| {py:mod}`emu_pk.validate` | shape and derivative error against CLASS |
