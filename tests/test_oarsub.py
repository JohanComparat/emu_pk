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
