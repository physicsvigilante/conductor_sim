"""Verify the saved Q35 teaching page with offline Chromium at two widths."""
import argparse
from math import isclose
from pathlib import Path

from playwright.sync_api import sync_playwright

from conductor_sim.model import EPSILON_0
from tools.jee_tunnel_profile import generate


def check(out):
    out = Path(out)
    page_path = out / "q35.html"
    generate(page_path)
    base = 1e-9 * 2**2 / (8*EPSILON_0)
    expected = [
        ["Sphere alone", "50%", f"{base:.2f} J", "0.00 J"],
        ["Sphere with axial bias", "25%", f"{base/4:.2f} J", f"{-2*base:.2f} J"],
    ]
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch()
        for width in (360, 1000):
            page = browser.new_page(viewport={"width": width, "height": 900}, java_script_enabled=False,
                                    offline=True, device_scale_factor=1)
            page.goto(page_path.resolve().as_uri())
            assert page.get_by_role("heading", name="Energy inside the tunnel", exact=True).is_visible()
            assert page.get_by_text("Finite-time passage requires", exact=False).is_visible()
            assert page.get_by_role("img").is_visible()
            for row, want in zip(page.locator("tbody tr").all(), expected, strict=True):
                cells = row.locator("th, td").all()
                assert all(c.is_visible() for c in cells)
                assert [c.inner_text() for c in cells] == want
            # Check the actual plotted vertices against independent force integrals.
            # Both curves use the same scale from the biased exit to the unbiased maximum.
            for j, line in enumerate(page.locator("polyline").all()):
                points = [tuple(map(float, pair.split(","))) for pair in line.get_attribute("points").split()]
                assert len(points) == 81
                for i, (x, y) in enumerate(points):
                    s = i / 80
                    change = 4*base*s*(1-s) - j*2*base*s
                    assert isclose(x, 76+376*s, abs_tol=0.006)
                    assert isclose(y, 36+146*(base-change)/(3*base), abs_tol=0.006)
            assert page.locator("polyline").count() == 2
            assert page.locator("polyline").nth(1).get_attribute("stroke-dasharray") == "7 4"
            markers = page.locator("svg circle").all()
            for marker, s, value in zip(markers, (0.5, 0.25), (base, base/4), strict=True):
                assert isclose(float(marker.get_attribute("cx")), 76+376*s, abs_tol=0.006)
                assert isclose(float(marker.get_attribute("cy")), 36+146*(base-value)/(3*base), abs_tol=0.006)
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            page.screenshot(path=str(out / f"q35-{width}.png"), full_page=True)
            page.close()
        browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    check(parser.parse_args().output)
