"""Shared drawing helpers for the design-document figures (diagram style, light theme)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import matplotlib.colors as mc

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e4e3df"

C = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100",
         magenta="#d55181", green="#008300", violet="#4a3aa7", red="#e34948", gray="#8a8984")

LINE_H = 0.205   # inches per body line at 8.6pt
TITLE_H = 0.30


def tint(color, amt=0.88):
    r, g, b = mc.to_rgb(color)
    return (r + (1 - r) * amt, g + (1 - g) * amt, b + (1 - b) * amt)


def canvas(w, h, title=None, subtitle=None):
    fig = plt.figure(figsize=(w, h), dpi=200, facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off"); ax.set_facecolor(SURFACE)
    if title:
        ax.text(0.35, h - 0.42, title, fontsize=17, fontweight="bold", color=INK, va="top")
    if subtitle:
        ax.text(0.35, h - 0.86, subtitle, fontsize=10.5, color=INK2, va="top")
    return fig, ax


def card(ax, x, ytop, w, title, lines=(), color="#2a78d6", fs=8.6, tfs=9.6, dashed=False,
         fill_amt=0.90, mono=False, h=None, title_color=INK):
    """Rounded card with a coloured left accent bar. Returns bottom y."""
    n = len(lines)
    hh = h if h is not None else (TITLE_H + 0.12 + n * LINE_H + (0.08 if n else -0.02))
    y = ytop - hh
    box = FancyBboxPatch((x, y), w, hh, boxstyle="round,pad=0,rounding_size=0.08",
                         fc=tint(color, fill_amt) if not dashed else SURFACE,
                         ec=color, lw=1.1, ls=(0, (4, 3)) if dashed else "-", zorder=2)
    ax.add_patch(box)
    ax.add_patch(Rectangle((x, y + 0.04), 0.06, hh - 0.08, fc=color, ec="none", zorder=3))
    ax.text(x + 0.16, ytop - 0.09, title, fontsize=tfs, fontweight="bold", color=title_color,
            va="top", zorder=4)
    for i, ln in enumerate(lines):
        ax.text(x + 0.16, ytop - TITLE_H - 0.07 - i * LINE_H, ln.lstrip("!"), fontsize=fs, color=INK2 if not ln.startswith("!") else INK,
                va="top", zorder=4, family="monospace" if mono else None)
    return y


def lane_header(ax, x, y, w, label, color, h=0.48):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                                fc=tint(color, 0.78), ec=color, lw=1.2, zorder=2))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10.5,
            fontweight="bold", color=INK, zorder=3)


def arrow(ax, p1, p2, color=INK2, lw=1.3, style="-|>", ls="-", rad=0.0, ms=11, z=5):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=ms, color=color, lw=lw,
                        linestyle=ls, connectionstyle=f"arc3,rad={rad}", zorder=z,
                        shrinkA=2, shrinkB=2)
    ax.add_patch(a)


def chevron(ax, x, y, color=MUTED, size=0.22):
    ax.add_patch(plt.Polygon([[x, y + size], [x + size * 0.9, y], [x, y - size]],
                             closed=True, fc=color, ec="none", zorder=5))


def footnote(ax, w, text, y=0.22):
    ax.text(0.35, y, text, fontsize=8.2, color=MUTED, va="bottom")


def save(fig, path):
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
