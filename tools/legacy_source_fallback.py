"""Static alternatives for the three source-backed legacy viewers in issue #111."""
from html import escape


def with_static_summary(html, *, hide, summary):
    """Keep the source banner and static results available without scripting."""
    if html.count("</head>") != 1 or html.count("<main>") != 1:
        raise ValueError("expected one document head and viewer main region")
    style = f"<noscript><style>{hide} {{ display: none !important; }}</style></noscript>"
    notice = (
        '<noscript><section class="card wide" id="static-source-summary">'
        '<h2>Static results</h2><p>' + escape(summary) + '</p></section></noscript>'
    )
    return html.replace("</head>", style + "</head>").replace("<main>", "<main>" + notice)
