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
        assert "N_EMU_SHARDS" in src
        # The budget is jobs *already waiting* plus the new array, not the
        # array alone -- a 94-element submission passed an array-only check and
        # still bounced, because the queue was not empty.
        assert "oarstat" in src and "Waiting" in src, (
            "the guard must ask the queue how full it is, not assume it empty")
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


class TestEveryJobParameterTravelsAsAnArgument:
    r"""OAR does not propagate the submitting shell's environment to the node.

    This has now bitten three times in one campaign, each silently:

    * the **mode** -- documented at the top of `run_generate.sh`, and the reason
      the other two were recognised at all;
    * the **design** (`EMU_N_TOTAL`, `EMU_PER_SHARD`) -- the submitter sized the
      array from 16000 and the workers generated from 150000;
    * the **arm** for training -- `EMU_ARM=f ... train` assembled the *curved*
      shards and wrote the *curved* weights, under a submission that printed
      `f`.

    Every one of them defaulted to something plausible instead of failing, which
    is why none was caught by running the thing.  So the rule is checked
    structurally: whatever a job needs in order to know *which* work it is
    doing has to appear inside the `-S` string.
    """

    @pytest.mark.skipif(not shutil.which("bash"), reason="needs bash")
    def test_every_generate_submission_names_arm_and_design(self):
        for line in (OARSUB / "submit_campaign.sh").read_text().splitlines():
            if "run_generate.sh emu" not in line:
                continue
            for var in ("${ARM}", "${EMU_N_TOTAL}", "${EMU_PER_SHARD}"):
                assert var in line, (
                    f"{var} is missing from a submission that would then use "
                    f"the node's default: {line.strip()}")

    @pytest.mark.skipif(not shutil.which("bash"), reason="needs bash")
    def test_the_training_submission_names_the_arm(self):
        src = (OARSUB / "submit_campaign.sh").read_text()
        args = [l for l in src.splitlines() if l.startswith("TRAIN_ARGS=")]
        assert args, "TRAIN_ARGS is not defined"
        assert "${ARM}" in args[0], (
            f"the arm is missing, so a training job silently trains arm 'c': "
            f"{args[0].strip()}")

    @pytest.mark.skipif(not shutil.which("bash"), reason="needs bash")
    def test_the_arm_lands_in_the_position_run_train_reads(self):
        """Positional, and `run_train.sh` shifts past it to reach the flags."""
        script = (
            'EPOCHS=240; TAG=t; ARM=f; TRAIN_FLAGS="--no-schedule --no-reduced"\n'
            'TRAIN_ARGS="${EPOCHS} ${TAG} ${ARM} ${TRAIN_FLAGS}"\n'
            'set -- $TRAIN_ARGS\n'
            'a="$1|$2|$3"\n'
            'if [ $# -gt 3 ]; then shift 3; else set --; fi\n'
            'echo "$a|$*"\n')
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        assert r.stdout.strip() == "240|t|f|--no-schedule --no-reduced"

    @pytest.mark.skipif(not shutil.which("bash"), reason="needs bash")
    def test_no_job_script_reads_a_campaign_variable_it_was_not_given(self):
        """`run_generate.sh` must prefer its arguments over the defaults.

        The defaults exist for a login-node run and must stay; what must not
        happen is a job silently using them when the submitter named something
        else.
        """
        src = (OARSUB / "run_generate.sh").read_text()
        assert 'EMU_N_TOTAL="${3:-${EMU_N_TOTAL}}"' in src
        assert 'EMU_PER_SHARD="${4:-${EMU_PER_SHARD}}"' in src


    @pytest.mark.skipif(not shutil.which("bash"), reason="needs bash")
    def test_the_queue_guard_survives_an_empty_queue(self):
        """`grep -c` prints 0 and exits 1 when it matches nothing.

        So `count=$(... | grep -c X || echo 0)` yields "0\n0", and the
        arithmetic that follows dies with a syntax error -- precisely when the
        queue is empty, which is the one case the guard should wave through.
        That is not hypothetical: it refused both control arms the first time
        the queue drained.
        """
        # Under the real `set -euo pipefail`, which is what the file uses.
        script = (
            'set -euo pipefail\n'
            '_waiting=$(echo "no matches here" | grep -c Waiting) || _waiting=0\n'
            '_room=$(( 100 - _waiting ))\n'
            'echo "$_room"\n')
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip() == "100"

    @pytest.mark.skipif(not shutil.which("bash"), reason="needs bash")
    def test_the_two_wrong_repairs_really_are_wrong(self):
        """Both were tried, and both broke only on an empty queue.

        `|| echo 0` inside the substitution yields "0\\n0"; a `${n:-0}` default
        *after* the assignment never runs, because `set -e` has already killed
        the script.  Pinned so neither comes back looking reasonable.
        """
        run = lambda body: subprocess.run(
            ["bash", "-c", "set -euo pipefail\n" + body],
            capture_output=True, text=True)

        r = run('n=$(echo x | grep -c Waiting || echo 0)\necho "[$n]"\n')
        assert r.stdout.strip() == "[0\n0]", "the doubling is the point"

        r = run('n=$(echo x | grep -c Waiting)\nn=${n:-0}\necho "[$n]"\n')
        assert r.returncode != 0 and not r.stdout, (
            "set -e should kill this before the default is applied")

    @pytest.mark.skipif(not shutil.which("bash"), reason="needs bash")
    def test_the_guard_does_not_use_the_broken_idiom(self):
        src = (OARSUB / "submit_campaign.sh").read_text()
        assert "grep -c Waiting || echo 0" not in src, (
            "that idiom yields '0\\n0' on an empty queue")
