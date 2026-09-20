"""Regenerate the corrected Wang Lang death-probability charts as SVG."""

from math import comb
from pathlib import Path


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "assets" / "img" / "sgs"
WIDTH, HEIGHT = 640, 480
LEFT, RIGHT, TOP, BOTTOM = 76, 606, 42, 410
COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c")


def effective_rank(t: int, x: int) -> int:
    """Wang Lang's comparison rank after Jici, capped at K (13)."""
    return min(13, t + x if t < x else t)


def death_probability(x: int, n: int) -> float:
    """Return P(K >= 7-x), averaging the conditional binomial tail over T."""
    needed = 7 - x
    if needed > n:
        return 0.0

    total = 0.0
    for t in range(1, 14):
        q = (14 - effective_rank(t, x)) / 13
        total += sum(
            comb(n, k) * q**k * (1 - q) ** (n - k)
            for k in range(needed, n + 1)
        )
    return total / 13


def sx(x: float) -> float:
    return LEFT + (RIGHT - LEFT) * (x + 0.35) / 6.7


def sy(y: float, ymax: float) -> float:
    return BOTTOM - (BOTTOM - TOP) * y / ymax


def base_svg(ymax: float, ystep: float) -> list[str]:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="DejaVu Sans, Arial, sans-serif" font-size="14" fill="#111">',
    ]

    tick = 0.0
    while tick <= ymax + 1e-12:
        y = sy(tick, ymax)
        parts.append(
            f'<line x1="{LEFT}" y1="{y:.2f}" x2="{RIGHT}" y2="{y:.2f}" '
            'stroke="#d8d8d8" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{LEFT - 10}" y="{y + 5:.2f}" text-anchor="end">{tick:.1f}</text>'
        )
        tick += ystep

    parts.extend(
        [
            f'<line x1="{LEFT}" y1="{TOP}" x2="{LEFT}" y2="{BOTTOM}" stroke="#111"/>',
            f'<line x1="{LEFT}" y1="{BOTTOM}" x2="{RIGHT}" y2="{BOTTOM}" stroke="#111"/>',
        ]
    )
    for x in range(7):
        px = sx(x)
        parts.append(
            f'<line x1="{px:.2f}" y1="{BOTTOM}" x2="{px:.2f}" y2="{BOTTOM + 5}" '
            'stroke="#111"/>'
        )
        parts.append(
            f'<text x="{px:.2f}" y="{BOTTOM + 24}" text-anchor="middle">{x}</text>'
        )
    parts.extend(
        [
            f'<text x="{(LEFT + RIGHT) / 2:.2f}" y="460" text-anchor="middle" '
            'font-style="italic">x</text>',
            f'<text x="20" y="{(TOP + BOTTOM) / 2:.2f}" text-anchor="middle" '
            f'transform="rotate(-90 20 {(TOP + BOTTOM) / 2:.2f})">'
            'P(K ≥ 7 − x)</text>',
        ]
    )
    return parts


def finish_svg(parts: list[str], filename: str) -> None:
    parts.extend(["</g>", "</svg>"])
    (OUTPUT_DIR / filename).write_text("\n".join(parts) + "\n", encoding="utf-8")


def plot_n3() -> None:
    ymax = 0.8
    parts = base_svg(ymax, 0.1)
    color = COLORS[0]
    for x in range(7):
        value = death_probability(x, 3)
        px, py = sx(x), sy(value, ymax)
        parts.append(
            f'<line x1="{px:.2f}" y1="{BOTTOM}" x2="{px:.2f}" y2="{py:.2f}" '
            f'stroke="{color}" stroke-width="2"/>'
        )
        parts.append(
            f'<circle cx="{px:.2f}" cy="{py:.2f}" r="5" fill="{color}">'
            f'<title>x={x}, n=3, probability={value:.6f}</title></circle>'
        )
    parts.append(
        '<text x="590" y="68" text-anchor="end" font-size="13" fill="#555">n = 3</text>'
    )
    finish_svg(parts, "gushe2.svg")


def plot_all_n() -> None:
    ymax = 0.8
    parts = base_svg(ymax, 0.1)
    offsets = (-0.12, 0.0, 0.12)

    for n, color, offset in zip(range(1, 4), COLORS, offsets):
        for x in range(7):
            value = death_probability(x, n)
            px, py = sx(x + offset), sy(value, ymax)
            parts.append(
                f'<line x1="{px:.2f}" y1="{BOTTOM}" x2="{px:.2f}" y2="{py:.2f}" '
                f'stroke="{color}" stroke-width="2"/>'
            )
            parts.append(
                f'<circle cx="{px:.2f}" cy="{py:.2f}" r="4.5" fill="{color}">'
                f'<title>x={x}, n={n}, probability={value:.6f}</title></circle>'
            )

    for index, (n, color) in enumerate(zip(range(1, 4), COLORS)):
        y = 58 + 23 * index
        parts.append(f'<line x1="516" y1="{y}" x2="538" y2="{y}" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<circle cx="527" cy="{y}" r="4" fill="{color}"/>')
        parts.append(f'<text x="546" y="{y + 5}" font-size="13">n = {n}</text>')
    finish_svg(parts, "gushe2_n.svg")


if __name__ == "__main__":
    plot_n3()
    plot_all_n()
