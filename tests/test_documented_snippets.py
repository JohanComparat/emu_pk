"""Every fenced ``python`` block in the README and the docs is executed here.

Documentation drifts silently: a snippet that stops working still renders, and
a reader meets the failure before anyone else does.  The blocks are run in file
order in a shared namespace, because a tutorial continues from what it has
already defined -- ``pk_cb = emu.pk_cb(k, 0.0, theta)`` is a whole block, and it
needs the ``emu`` and ``theta`` from the top of the page.

A block that documents a failure is executed too, and asserted to raise: a
trailing ``# ValueError: ...`` comment is the marker, which is also how the
block reads to a human.
"""
from __future__ import annotations

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
FENCE = re.compile(r"^```python\n(.*?)^```", re.M | re.S)
RAISES = re.compile(r"^\s*#\s*(ValueError|TypeError|KeyError|IndexError)", re.M)


def _pages():
    docs = ROOT / "docs"
    pages = [ROOT / "README.md"]
    if docs.is_dir():
        pages += sorted(p for p in docs.rglob("*.md") if "_build" not in p.parts)
    return [p for p in pages if p.is_file() and FENCE.search(p.read_text())]


PAGES = _pages()


def test_there_are_snippets_to_check():
    """A regex that silently matches nothing would make every test below pass."""
    assert PAGES, "no documentation page with a ```python block was found"
    n = sum(len(FENCE.findall(p.read_text())) for p in PAGES)
    assert n >= 15, f"only {n} snippets found; the fence pattern may have drifted"


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_the_snippets_on_this_page_run(page):
    text = page.read_text()
    ns: dict = {}
    for m in FENCE.finditer(text):
        body = m.group(1)
        where = f"{page.relative_to(ROOT)}:{text[:m.start()].count(chr(10)) + 1}"
        want = RAISES.search(body)
        code = compile(body, where, "exec")
        if want is None:
            exec(code, ns)                      # noqa: S102 -- that is the point
            continue
        with pytest.raises(Exception) as exc:   # noqa: PT011
            exec(code, ns)                      # noqa: S102
        assert type(exc.value).__name__ == want.group(1), (
            f"{where} documents {want.group(1)} and raised "
            f"{type(exc.value).__name__}")
