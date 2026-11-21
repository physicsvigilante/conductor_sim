"""Exercise the source-backed p2, p26 and p27 exports in actual offline Chromium."""
from math import isclose

import pytest
from playwright.sync_api import sync_playwright

from tools.m36b_jee_replay import _event_record
from tools.m36c_jee_conductor_replay import _p27_original_A_B
from tools.m36d_true_solid_sphere import run_all as render_solid


@pytest.fixture(scope="module")
def pages(tmp_path_factory):
    folder = tmp_path_factory.mktemp("legacy-source-viewers")
    sheet = _event_record("sheet-stop", folder)
    solid = render_solid(folder)
    shells = _p27_original_A_B(folder)
    return {name: folder / record["viewer_file"] for name, record in (
        ("sheet", sheet), ("solid", solid), ("shells", shells)
    )}


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        yield browser
        browser.close()


@pytest.mark.parametrize("name", ["sheet", "solid", "shells"])
@pytest.mark.parametrize("width", [375, 1280])
@pytest.mark.parametrize("js", [False, True], ids=["no-js", "js"])
def test_legacy_source_view(browser, pages, name, width, js, tmp_path):
    context = browser.new_context(viewport={"width": width, "height": 820},
                                  java_script_enabled=js, offline=True)
    try:
        page = context.new_page()
        errors, external = [], []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("request", lambda req: external.append(req.url)
                if req.url.startswith(("https:", "http:")) else None)
        page.goto(pages[name].resolve().as_uri())
        body = page.locator("body").inner_text()
        notice = page.locator("#static-source-summary")
        controls = page.locator("input:visible, button:visible, select:visible")
        if js:
            assert not notice.count() and controls.count()
            slider = page.locator("#frame")
            if name == "sheet":
                initial_readout = page.locator("#readout").inner_text()
            slider.focus()
            slider.press("End")
            assert int(slider.input_value()) > 0
            if name == "solid":
                assert "TEACHER HYPOTHETICAL" in page.locator("[data-m36e-state]:visible").inner_text()
            elif name == "shells":
                assert "TEACHER" in page.locator("#case").inner_text()
            else:
                final_readout = page.locator("#readout").inner_text()
                assert initial_readout.strip() and final_readout.strip() and initial_readout != final_readout
        else:
            assert notice.is_visible() and "require JavaScript" in notice.inner_text()
            assert controls.count() == 0 and page.locator("canvas:visible").count() == 0
            if name == "sheet":
                assert "0.44271 mm" in notice.inner_text() and "launch point" in notice.inner_text()
                assert "choice 1, 0.44 mm" in body and "turns 0.55729 mm" in body
                assert "100 eV" in body and "teacher-chosen" in body and "stop:x" in body
            elif name == "solid":
                answers = page.locator("[data-m36e-state]:visible")
                assert answers.count() == 1 and "TEACHER numerical example" in answers.inner_text()
                assert "V(R/2) = kQ/R" in notice.inner_text()
                for key, volts in (("V_half_R", 89.875517923), ("V_three_half_R", 59.917011948)):
                    value = answers.locator(f'[data-m36e-number="{key}"]').inner_text().split()
                    assert value[1] == "V" and isclose(float(value[0]), volts, rel_tol=1e-9)
            else:
                assert "V(A) = 539.251535 V" in body and "V(B) = 314.563639 V" in body
                for given in ("central q = 1 nC", "inner shell q₁ = 2 nC", "outer shell q₂ = 3 nC",
                              "a = 0.10 m", "b = 0.20 m", "cavity probe at r=0.04m", "gap probe at r1=0.15m"):
                    assert given in body
                assert "TEACHER choices" in body and "thin shells" in notice.inner_text()
        if name == "solid":
            assert "Notes p26 Illustration 1" in page.title()
            assert "Notes p26 Illustration 1" in page.get_by_role("heading", level=1).inner_text()
            assert "JEE p26" not in body and "JEE original p26" not in body
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
        assert not errors and not external
        page.screenshot(path=str(tmp_path / f"{name}-{width}-js{int(js)}.png"), full_page=True)
    finally:
        context.close()
