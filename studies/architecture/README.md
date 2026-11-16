# Architecture scan

This study measures how the test error of the emulator depends on the depth and the width of the network and on the number of training cosmologies.
No theorem gives the optimal architecture for this problem: the universal approximation theorems guarantee that a network exists, not its size, and the training is non-convex.
We therefore measure the error on the production training set and compare it with the error of the training truth itself, 0.039 % from CAMB's converged answer in the median ([reproducing.md](../../docs/reproducing.md)).

`scan.py` trains one network and writes its scores; `run_scan.sh` runs the scan; `plot.py` draws the figure.

## Set-up

- Data: `work/v21/training_set_v21_c.npz`, 149 975 CAMB cosmologies at 31 redshifts.
- Split by cosmology, fixed for every run: 3 000 test, 1 500 validation, and a training pool of 145 475.
  A run trains on the first `--n-cos` cosmologies of the pool, so the training sets are nested.
- Target, standardisation and loss as in `emu_pk.train`: $\ln P$ with the primordial power law divided out, mean squared error in $\ln P$.
- Adam, warm-up and cosine decay over 200 000 steps of 1 024 rows, peak learning rate $10^{-3}$; the weights of the best validation loss are kept.
- Score: per test cosmology, the maximum over $10^{-3} < k < 10\,h\,{\rm Mpc}^{-1}$ of $|\Delta \ln P_{\rm m}|$ at $z = 0$, then its median over the 3 000 cosmologies.

## Runs

| training cosmologies | hidden layers |
|---|---|
| 145 475 | [256], [512], [1024], 2×256, 2×512, 2×1024, 4×256, 4×512, 4×1024, 6×256, 6×512, 6×1024 |
| 36 000, 9 000, 2 250 | [1024], 2×512, 4×256, 4×512, 4×1024, 6×512 |

The step budget is the same for every run, so the full pool sees 45 epochs and the shipped network, trained for 237 epochs, is not one of the runs.

```bash
OUT=~/data/emu_pk_architecture_scan bash studies/architecture/run_scan.sh
python studies/architecture/plot.py --scan ~/data/emu_pk_architecture_scan --out arch_scan.pdf
```

## Activation

`scan.py --act` replaces the learned activation by a fixed one; the default, `cp`, is the activation `emu_pk` ships, $[\gamma + (1-\gamma)\,\sigma(\beta x)]\,x$ with $\beta$ and $\gamma$ learned per unit.
We train the shipped architecture, 4×512 on the full pool, with ReLU, SiLU, GELU, tanh, softplus and sigmoid, from the same initial weights (without $\beta$ and $\gamma$).
SiLU is `cp` with $\beta = 1$ and $\gamma = 0$ fixed.
Each run also records the roughness of $\partial\ln P/\partial\omega_{\rm b}$ at $k = 0.1\,h\,{\rm Mpc}^{-1}$ and $z = 0$: the root mean square of its second difference along 400 values of $\omega_{\rm b}$, over that of the derivative.

```bash
python studies/architecture/scan.py --hidden 512 512 512 512 --n-cos 145475 --act relu \
    --out ~/data/emu_pk_architecture_scan
```
