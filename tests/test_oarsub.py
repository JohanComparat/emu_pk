"""The cluster scripts parse.

Nothing here runs a job -- OAR is not available off the cluster and these
scripts are mostly `oarsub` invocations.  What *is* checkable off the cluster is
that they are syntactically valid bash, and that turns out to be worth a test:
a missing `fi` in `run_train.sh` reached the cluster and cost a submitted job,
because the check that should have caught it was written

    bash -n oarsub/_campaign_env.sh oarsub/run_train.sh

and `bash -n` parses only its *first* argument.  Everything after it becomes a
positional parameter of the script being parsed, so the second file was never
looked at and the check passed.  One file per invocation, here, in a loop.
"""
import pathlib
import shutil
import subprocess

import pytest

OARSUB = pathlib.Path(__file__).resolve().parent.parent / "oarsub"
SCRIPTS = sorted(OARSUB.glob("*.sh"))


def test_there_are_scripts_to_check():
    """A glob that matches nothing makes every test below vacuously pass."""
    assert SCRIPTS, f"no *.sh under {OARSUB}"


@pytest.mark.skipif(not shutil.which("bash"), reason="needs bash")
@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_script_is_valid_bash(script):
    r = subprocess.run(["bash", "-n", str(script)],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"{script.name}:\n{r.stderr}"


@pytest.mark.skipif(not shutil.which("bash"), reason="needs bash")
def test_bash_n_only_checks_its_first_argument(tmp_path):
    """The trap itself, pinned, so the loop above is not quietly rewritten
    back into one invocation by someone tidying it up."""
    good, bad = tmp_path / "good.sh", tmp_path / "bad.sh"
    good.write_text("echo fine\n")
    bad.write_text("if true; then\n  echo x\n")          # no `fi`

    assert subprocess.run(["bash", "-n", str(bad)]).returncode != 0
    # ...and yet, passed second, it is not examined at all:
    assert subprocess.run(["bash", "-n", str(good), str(bad)]).returncode == 0


class TestTheDesignArmSelectsItsOwnPaths:
    """`campaign_arm` is the one piece of these scripts with logic in it.

    A campaign runs the nine-parameter design and its flat control at the same
    time.  If they shared a shard directory the second would find the first's
    files, skip on filename -- the name carries the shard index and the design
    offset, not the box -- and report success having solved nothing.  So every
    path the two arms touch has to differ, and that is worth checking off the
    cluster, where the consequence is a failed test rather than a wasted
    allocation.
    """

    def _arm(self, arm):
        """Source the campaign env, apply the arm, print what it decided."""
        script = (
            'set -u\n'
            'export EMU_PK_PROJECT=test-project EMU_PK_WORK=/tmp/w\n'
            f'source {OARSUB}/_campaign_env.sh >/dev/null 2>&1\n'
            f'campaign_arm {arm} || exit 3\n'
            'echo "$EMU_PK_ARM|$EMU_PK_SHARDS_EMU|$EMU_PK_DATASET'
            '|$EMU_PK_WEIGHTS|$EMU_PIN"\n')
        r = subprocess.run(["bash", "-c", script], capture_output=True,
                           text=True)
        return r.returncode, r.stdout.strip()

    @pytest.mark.skipif(not shutil.which("bash"), reason="needs bash")
    def test_the_two_arms_share_no_path(self):
        _, c = self._arm("c")
        _, f = self._arm("f")
        cp, fp = c.split("|"), f.split("|")
        for i, what in enumerate(("arm", "shards", "dataset", "weights")):
            assert cp[i] != fp[i], f"both arms use the same {what}: {cp[i]}"

    @pytest.mark.skipif(not shutil.which("bash"), reason="needs bash")
    def test_only_the_flat_arm_pins(self):
        assert self._arm("c")[1].split("|")[4] == ""
        assert self._arm("f")[1].split("|")[4] == "--pin Omega_k=0"

    @pytest.mark.skipif(not shutil.which("bash"), reason="needs bash")
    def test_neither_arm_claims_the_1_0_0_directory(self):
        """`shards_emu` without a suffix is the reproduction of the shipped
        model.  A run that wrote there would mix two boxes into it."""
        for arm in ("c", "f"):
            shards = self._arm(arm)[1].split("|")[1]
            assert not shards.endswith("/shards_emu")

    @pytest.mark.skipif(not shutil.which("bash"), reason="needs bash")
    def test_an_unknown_arm_is_refused(self):
        """Rather than silently producing a fifth set of paths."""
        assert self._arm("zzz")[0] == 3

    @pytest.mark.skipif(not shutil.which("bash"), reason="needs bash")
    def test_the_array_cap_is_enforced_before_the_scheduler_sees_it(self):
        """GRICAD refuses more than 100 waiting jobs, and an array of N is N
        jobs.  The submitter used to hand that rejection to the scheduler,
        which reports it without saying which knob to turn."""
        src = (OARSUB / "submit_campaign.sh").read_text()
        assert "N_EMU_SHARDS" in src and "-gt 94" in src
        assert "EMU_PER_SHARD to" in src, "the message must name the fix"


class TestTheDesignReachesTheWorker:
    r"""OAR does not propagate the submitting shell's environment.

    `submit_campaign.sh` reads `EMU_N_TOTAL` on the frontend to size the array,
    and the job re-reads it on the node from `_campaign_env.sh` -- where it gets
    the *default*.  So `EMU_N_TOTAL=16000 submit_campaign.sh emu` submitted
    sixteen array elements that each generated from the 150000-point design.

    Nothing downstream could see it.  The parameter names match, so the shard
    stamp passes; the grids match, so `assemble` passes; only the cosmologies
    are from a design nobody asked for, in files named as though they were from
    the one that was.  The fix is to pass the design as an argument, and this
    test is what stops it drifting back.
    """

    @pytest.mark.skipif(not shutil.which("bash"), reason="needs bash")
    def test_the_submitter_passes_the_design_to_the_job(self):
        src = (OARSUB / "submit_campaign.sh").read_text()
        for line in src.splitlines():
            if "run_generate.sh emu" in line:
                assert "${EMU_N_TOTAL}" in line and "${EMU_PER_SHARD}" in line, (
                    f"this submission does not carry the design: {line.strip()}")

    @pytest.mark.skipif(not shutil.which("bash"), reason="needs bash")
    def test_the_worker_prefers_its_argument_over_the_default(self):
        """The whole point: an argument must beat `_campaign_env.sh`."""
        script = (
            'set -u\n'
            'export EMU_PK_PROJECT=test-project EMU_PK_WORK=/tmp/w\n'
            'set -- emu c 16000 1000\n'
            f'source {OARSUB}/_campaign_env.sh >/dev/null 2>&1\n'
            'EMU_N_TOTAL="${3:-${EMU_N_TOTAL}}"\n'
            'EMU_PER_SHARD="${4:-${EMU_PER_SHARD}}"\n'
            'echo "$EMU_N_TOTAL|$EMU_PER_SHARD"\n')
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        assert r.stdout.strip() == "16000|1000"

    @pytest.mark.skipif(not shutil.which("bash"), reason="needs bash")
    def test_it_still_falls_back_when_nothing_is_named(self):
        script = (
            'set -u\n'
            'export EMU_PK_PROJECT=test-project EMU_PK_WORK=/tmp/w\n'
            'set -- emu c\n'
            f'source {OARSUB}/_campaign_env.sh >/dev/null 2>&1\n'
            'EMU_N_TOTAL="${3:-${EMU_N_TOTAL}}"\n'
            'echo "$EMU_N_TOTAL"\n')
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        assert r.stdout.strip() == "150000"
