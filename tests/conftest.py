import os

# float64 before jax is imported anywhere: several of these tests compare an
# autodiff derivative against a central difference at the 1e-7 level, which
# float32 cannot resolve -- the test would fail for a reason that has nothing to
# do with what it is testing.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

LEGACY = os.path.join(os.path.dirname(__file__), "..", "emu_pk", "data",
                      "class_nu_ratio_legacy.npz")
