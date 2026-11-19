# Matrix-product precision on a GPU

JAX multiplies float32 matrices in TF32, with ten mantissa bits, on cards that have it (A100, H100), unless `JAX_DEFAULT_MATMUL_PRECISION=highest`.
`precision_check.py` evaluates $\ln P_{\rm m}$ for 1 000 cosmologies of `box.sample(1000, seed=7)`, at 200 wavenumbers in $10^{-4} < k < 300\,h\,{\rm Mpc}^{-1}$ and the 31 training redshifts, on the CPU and on the GPU at the default and at the highest precision, and reports the difference.

| card | default: max | default: median | `highest`: max | `highest`: median |
|---|---|---|---|---|
| NVIDIA V100 SXM2 (GRICAD, Bigfoot) | 2.3e-5 | 0 | 2.3e-5 | 0 |
| NVIDIA A100 PCIe (GRICAD, Bigfoot) | 6.9e-3 | 2.7e-4 | 3.8e-5 | 4.8e-7 |

The entries are $|\Delta \ln P|$ against the CPU; `precision_v100.json` and `precision_a100.json` hold them with the 99th percentiles.
At the default, the A100 moves $\ln P$ by up to 0.69 %, more than the network's own largest error (0.18 %); at `highest` it agrees with the CPU to float32 rounding, as the V100, which has no TF32, does at either setting.
On an A100 or H100, set `JAX_DEFAULT_MATMUL_PRECISION=highest` to evaluate the network as it was trained.

```bash
python studies/gpu_precision/precision_check.py --json precision_<card>.json   # on a GPU node
```
