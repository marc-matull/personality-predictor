"""Ocean theme of the app: animated waves, fish and bubbles.

Everything is pure CSS and inline HTML (the wave images are embedded as
data URIs), so no external files or additional packages are needed.
"""

from urllib.parse import quote

# One seamless wave tile (the curve starts and ends at the same height with
# the same slope, so repeating it horizontally shows no seam)
_WAVE_TILE = (
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1200 80' "
    "preserveAspectRatio='none'>"
    "<path d='M0 40 C200 0 400 0 600 40 S1000 80 1200 40 L1200 80 L0 80 Z' "
    "fill='{color}'/></svg>"
)


def _wave_url(color: str) -> str:
    """Return a CSS ``url(...)`` with an inline SVG wave of the given color.

    Args:
        color: Fill color of the wave, e.g. ``"#0277bd"``.

    Returns:
        A CSS ``url("data:image/svg+xml,...")`` expression.
    """
    return 'url("data:image/svg+xml,{}")'.format(
        quote(_WAVE_TILE.format(color=color))
    )


def get_ocean_css() -> str:
    """Return the CSS of the ocean theme (without the ``<style>`` tags).

    Returns:
        The stylesheet as a string without blank lines, which keeps it safe
        to embed through ``st.markdown``.
    """
    css = """
.stApp {
    background: linear-gradient(180deg, #e1f5fe 0%, #b3e5fc 40%,
                                #4fc3f7 80%, #0288d1 100%);
    background-attachment: fixed;
}
[data-testid="stForm"] {
    background: rgba(255, 255, 255, 0.78);
    border: 1px solid #81d4fa;
    border-radius: 16px;
}
h1, h2, h3 { color: #01579b; }
.ocean-banner {
    position: relative; height: 170px; overflow: hidden;
    border-radius: 16px; margin-bottom: 1rem;
    background: linear-gradient(180deg, #b3e5fc 0%, #29b6f6 100%);
}
.wave {
    position: absolute; left: 0; bottom: 0; width: 100%; height: 80px;
    background-repeat: repeat-x; background-size: 1200px 80px;
    animation: wave-move 12s linear infinite;
}
.wave-back {
    height: 95px; opacity: 0.55; animation-duration: 18s;
    animation-direction: reverse;
    background-image: WAVE_BACK;
}
.wave-front { background-image: WAVE_FRONT; }
@keyframes wave-move {
    from { background-position-x: 0; }
    to { background-position-x: 1200px; }
}
.fish {
    position: absolute; font-size: 2rem; left: -10%;
    animation: swim 16s linear infinite;
}
.fish-1 { top: 25%; animation-duration: 18s; }
.fish-2 { top: 50%; animation-duration: 24s; animation-delay: 4s; }
.fish-3 { top: 70%; animation-duration: 14s; animation-delay: 9s; }
@keyframes swim {
    from { left: -10%; }
    to { left: 110%; }
}
.bubble {
    position: absolute; bottom: -20px; width: 10px; height: 10px;
    border-radius: 50%; background: rgba(255, 255, 255, 0.7);
    animation: rise 7s ease-in infinite;
}
.bubble-1 { left: 12%; animation-delay: 0s; }
.bubble-2 { left: 38%; animation-delay: 2s; width: 14px; height: 14px; }
.bubble-3 { left: 63%; animation-delay: 4s; }
.bubble-4 { left: 85%; animation-delay: 1s; width: 7px; height: 7px; }
@keyframes rise {
    from { transform: translateY(0); opacity: 0.9; }
    to { transform: translateY(-190px); opacity: 0; }
}
.ocean-footer {
    margin-top: 2rem; padding: 1.2rem; text-align: center;
    color: #ffffff; font-size: 1.1rem;
    background: rgba(1, 87, 155, 0.65); border-radius: 16px;
}
"""
    css = css.replace("WAVE_BACK", _wave_url("#0277bd"))
    css = css.replace("WAVE_FRONT", _wave_url("#01579b"))
    # Remove blank lines, they would end the HTML block in Markdown
    return "\n".join(line for line in css.splitlines() if line.strip())


def get_banner_html() -> str:
    """Return the animated ocean banner (waves, fish and bubbles).

    Returns:
        HTML without blank lines.
    """
    return (
        '<div class="ocean-banner">'
        '<span class="fish fish-1">\U0001F41F</span>'
        '<span class="fish fish-2">\U0001F420</span>'
        '<span class="fish fish-3">\U0001F421</span>'
        '<span class="bubble bubble-1"></span>'
        '<span class="bubble bubble-2"></span>'
        '<span class="bubble bubble-3"></span>'
        '<span class="bubble bubble-4"></span>'
        '<div class="wave wave-back"></div>'
        '<div class="wave wave-front"></div>'
        "</div>"
    )


def get_footer_html() -> str:
    """Return the footer of the app.

    Returns:
        HTML without blank lines.
    """
    return (
        '<div class="ocean-footer">'
        "\U0001F30A Thanks for diving in! "
        "\U0001F41A \U0001F419 \U0001F980"
        "</div>"
    )
