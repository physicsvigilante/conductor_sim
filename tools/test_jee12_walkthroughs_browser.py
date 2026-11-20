"""Real Chromium regression for the student-visible field-work sign."""

from pathlib import Path

import pytest

from tools.check_jee12_walkthroughs_browser import check
from tools.jee12_walkthroughs import generate


def test_visible_field_work_sign_disagrees_with_correct_metadata(tmp_path):
    generate(Path(__file__).resolve().parents[1], tmp_path)
    page = tmp_path / "p1-external-work.html"
    html = page.read_text(encoding="utf-8")
    original = ">-40 eV</output>"
    assert html.count(original) == 1
    page.write_text(html.replace(original, ">+40 eV</output>"), encoding="utf-8")

    with pytest.raises(AssertionError) as failure:
        check(tmp_path)
    assert str(failure.value).startswith(
        "p1-external-work/desktop-js #p1-field-ev: expected '-40 eV', saw '+40 eV'"
    )
