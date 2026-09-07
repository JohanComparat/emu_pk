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


def shipped_weights_cover_the_box():
    """Whether ``emu_pk/data/emu_pk_mlp.npz`` was trained on *this* box.

    It is not, between a box growing and the retraining that follows, and that
    gap is deliberate: a checkpoint blind to a sampled parameter is refused
    rather than quietly ignoring it, so every test that loads the shipped file
    fails loudly the moment ``box.PARAMS`` grows.  That is the tripwire working.

    Those tests are skipped with a reason rather than deleted or forced through
    with ``allow_narrow_box``: they are the ones that will say whether the
    *retrained* weights are any good, and a suite that passed by pretending
    would have nothing left to say when the new file lands.
    """
    import numpy as np

    from emu_pk import box
    from emu_pk.model import ANALYTIC, DEFAULT_WEIGHTS

    if not DEFAULT_WEIGHTS.exists():
        return False, "no weights are shipped"
    with np.load(DEFAULT_WEIGHTS) as d:
        if "params_order" not in d.files:
            return True, ""
        order = [str(p) for p in d["params_order"]]
        reduced = str(d.get("target_form", "raw")) == "reduced"
    absent = [p for p in box.PARAMS
              if p not in order and not (reduced and p in ANALYTIC)]
    if absent:
        return False, (
            f"the shipped weights have no input for {absent}: they were "
            f"trained on a narrower box and this one samples it.  Retrain "
            f"(see docs/reproducing.md); until then there is nothing here "
            f"worth asserting about them.")
    return True, ""


def skip_if_shipped_weights_are_stale():
    """``pytest.skip`` unless the shipped file covers the current box."""
    import pytest

    ok, why = shipped_weights_cover_the_box()
    if not ok:
        pytest.skip(why)


@__import__("pytest").fixture
def _needs_current_shipped_weights():
    """Skip a test that asserts something about the *shipped* network."""
    skip_if_shipped_weights_are_stale()
