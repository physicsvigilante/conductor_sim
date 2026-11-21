"""Check workbook Q39/Q40/Q55 signed values, shading and native curves in offline Chromium."""
import argparse
from math import isclose
from pathlib import Path

from playwright.sync_api import sync_playwright

from tools.jee_field_area import generate


def _reference(number, x):
    if number == 55:
        return 2*x*x, -2*(x**3-1)/3
    if number == 39:
        return 50, -50*(x+2)
    if x <= 2:
        return -10*x, 10+5*x*x
    if x <= 4:
        return 20*(x-3), 40-10*(x-3)**2
    return 20, 30-20*(x-4)


def _points(element):
    return [tuple(map(float, pair.split(","))) for pair in element.get_attribute("points").split()]


def _check_paint(plot):
    # Playwright's layout visibility includes opacity-zero elements. Check the
    # paint channels and ancestors of every shape/label in the rendered SVG.
    hidden = plot.evaluate("""svg => {
        function painted(element) {
            for (let node = element; node; node = node.parentElement) {
                const style = getComputedStyle(node);
                if (Number(style.opacity) === 0 || style.display === "none" ||
                    style.visibility !== "visible") return false;
            }
            const style = getComputedStyle(element);
            const line = ["path", "polyline"].includes(element.localName);
            const color = line ? style.stroke : style.fill;
            if (color === "none" || color === "transparent" ||
                (line && parseFloat(style.strokeWidth) === 0) ||
                Number(line ? style.strokeOpacity : style.fillOpacity) === 0) return false;
            const context = document.createElement("canvas").getContext("2d");
            context.fillStyle = color;
            context.fillRect(0, 0, 1, 1);
            return context.getImageData(0, 0, 1, 1).data[3] > 0;
        }
        return [...svg.querySelectorAll("text,path,polyline,circle,polygon")]
            .filter(element => !painted(element)).map(element => element.localName);
    }""")
    assert not hidden, f"unpainted plot elements: {hidden}"


def _check_text(element):
    assert element.is_visible() and element.evaluate("""el => {
        if (!el.checkVisibility({opacityProperty: true, visibilityProperty: true})) return false;
        const context = document.createElement("canvas").getContext("2d");
        context.fillStyle = getComputedStyle(el).webkitTextFillColor;
        context.fillRect(0, 0, 1, 1);
        return context.getImageData(0, 0, 1, 1).data[3] > 0;
    }"""), "unpainted walkthrough text"


def _check_panel(page, number):
    panel = page.locator(f"#q{number}")
    expected = (["+100 V", "-100 V", "-100 V · option 4"] if number == 39 else
                ["-20 V", "+20 V", "+30 V · option 4"])
    if number == 55:
        expected = ["2.410 m", "1.410 m", "0.590 m", "-8.667 V", "-8.667 V"]
        answer = panel.get_by_text("x₁ > x₂ · option 2", exact=True)
        _check_text(answer)
        _check_text(panel.get_by_role("heading", name="Teacher example"))
        labels = panel.locator("dl dt").all()
        assert [label.inner_text() for label in labels] == [
            "Middle plane", "First spacing", "Second spacing",
            "First potential change", "Second potential change"]
        for label, value in zip(labels, expected):
            _check_text(label)
            sibling = label.locator("xpath=following-sibling::*[1][self::dd]")
            assert sibling.inner_text() == value
            _check_text(sibling)
    values = panel.locator("dd").all()
    assert all(v.is_visible() for v in values)
    assert [v.inner_text() for v in values] == expected
    start, end, target = (-2, 0, 0) if number == 39 else (0, 5, 2)
    ranges = ((0, 50), (-100, 0)) if number == 39 else ((-20, 20), (0, 40))
    if number == 55:
        start, end, target = 1, 3, 14**(1/3)
        ranges = ((0, 18), (-52/3, 0))
    plots = panel.get_by_role("img").all()
    assert len(plots) == 2
    for index, (plot, (low, high)) in enumerate(zip(plots, ranges)):
        assert plot.is_visible()
        _check_paint(plot)
        assert plot.locator("path").count() == 1
        zero = 28+100*high/(high-low)
        assert plot.locator("path").get_attribute("d") == f"M62 26 V130 M62 {zero:.2f} H392"
        assert plot.locator('text[y="17"]').text_content() == ("Axial field (V/m)" if index == 0 else "Potential (V)")
        assert plot.locator('text[y="172"]').text_content() == "x (m)"
        assert plot.locator('text[x="56"]').all_text_contents() == [f"{v:g}" for v in sorted({low, high, 0})]
        labels = plot.locator('text[y="151"]').all()
        assert all(label.is_visible() for label in labels)
        assert [label.text_content() for label in labels] == [str(i) for i in range(start, end+1)]
        points = _points(plot.locator("polyline"))
        assert len(points) == {39: 21, 40: 61, 55: 22}[number]
        for px, py in points:
            x = start + (px-62)*(end-start)/330
            v = _reference(number, x)[index]
            assert isclose(py, 28+100*(high-v)/(high-low), abs_tol=.02)
        assert points[0][0] == 62 and points[-1][0] == 392
        assert all(a[0] < b[0] for a,b in zip(points, points[1:]))
        marker = plot.locator("circle")
        target_px = 62+330*(target-start)/(end-start)
        assert isclose(float(marker.get_attribute("cx")), target_px, abs_tol=.006)
        assert isclose(float(marker.get_attribute("cy")), 28+100*(high-_reference(number,target)[index])/(high-low), abs_tol=.006)
        if index == 0:
            polygons = plot.locator("polygon").all()
            assert len(polygons) == (2 if number == 55 else 1)
            area = _points(polygons[0])
            zero = 28+100*high/(high-low)
            split = round(target_px, 2)
            assert area[0] == (62, zero) and area[-1] == (split, zero)
            assert area[1:-1] == [p for p in points if p[0] <= split]
            if number == 55:
                right = _points(polygons[1])
                assert right[0] == (split, zero) and right[-1] == (392, zero)
                assert right[1:-1] == [p for p in points if p[0] >= split]
                assert polygons[0].evaluate("e => getComputedStyle(e).fill") != polygons[1].evaluate("e => getComputedStyle(e).fill")
        else:
            assert plot.locator("polygon").count() == 0


def check(out):
    out = Path(out)
    path = out/"field-area.html"
    generate(path)
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch()
        for width in (360, 1000):
            page = browser.new_page(viewport={"width":width,"height":900}, java_script_enabled=False, offline=True)
            page.goto(path.resolve().as_uri())
            for number in (39, 40, 55):
                _check_panel(page, number)
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            page.screenshot(path=str(out/f"field-area-{width}.png"), full_page=True)
            if width == 360:
                mutations = [
                    (40, "#q40 svg {opacity:0}", "unpainted plot elements"),
                    (40, "#q40 polygon {fill:transparent}", "unpainted plot elements"),
                    (40, "#q40 circle {opacity:0}", "unpainted plot elements"),
                    (55, "#q55 dl {opacity:0}", "unpainted walkthrough text"),
                    (55, "#q55 strong {color:transparent}", "unpainted walkthrough text"),
                    (55, "#q55 h3 {color:transparent}", "unpainted walkthrough text"),
                    (55, "#q55 dt {display:none}", "unpainted walkthrough text"),
                ]
                for number, rule, reason in mutations:
                    style = page.evaluate_handle("""rule => {
                        const node = document.createElement("style");
                        node.textContent = rule;
                        document.head.appendChild(node);
                        return node;
                    }""", rule)
                    try:
                        _check_panel(page, number)
                    except AssertionError as exc:
                        if reason not in str(exc):
                            raise
                    else:
                        raise AssertionError(f"invisible content passed browser check: {rule}")
                    finally:
                        style.evaluate("node => node.remove()")
            page.close()
        browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    check(parser.parse_args().output)
