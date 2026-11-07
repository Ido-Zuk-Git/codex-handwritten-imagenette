#!/usr/bin/env python3
"""Render the val-accuracy-vs-iteration video for run 2, session 1.

Reads run/iterations.csv (iteration,train_acc,val_acc,minutes,primitive,note),
writes PNG frames to --frames-dir, then encodes an mp4 and a gif with ffmpeg.

    python3 scripts/make_video.py --csv run/iterations.csv --out media/val_vs_iteration
"""
import argparse
import csv
import glob
import os
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

# Palette (dataviz reference palette, light mode)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e2"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"  # series
RED = "#e34948"  # status: reverted

TITLE = "Run 2, session 1: hand-written Imagenette classifier (gpt-6.1-sol, 45 min)"
X_MIN, X_MAX, Y_MIN, Y_MAX = 0.5, 13.5, 20.0, 45.0
BELOW_IF_ABOVE = 42.0  # callout goes below the point when val > this

INTRO_S, GROW_S, HOLD_S, FINAL_S = 1.0, 0.35, 1.0, 2.5


def read_rows(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if not r.get("iteration"):
                continue
            rows.append(
                {
                    "i": int(r["iteration"]),
                    "val": float(r["val_acc"]),
                    "min": float(r["minutes"]),
                    "prim": (r.get("primitive") or "").strip(),
                    "reverted": "revert" in (r.get("primitive") or "").lower(),
                }
            )
    rows.sort(key=lambda r: r["i"])
    if not rows:
        sys.exit(f"no rows in {path}")
    return rows


def build_figure(w, h, dpi=100):
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi, facecolor=SURFACE)
    ax = fig.add_axes([0.075, 0.115, 0.895, 0.78])
    ax.set_facecolor(SURFACE)
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xticks(range(1, 14))
    ax.set_yticks(range(20, 46, 5))
    ax.tick_params(colors=INK_2, labelsize=11, length=0, pad=6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(1)
    ax.grid(True, axis="y", color=GRID, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    ax.set_xlabel("iteration", color=INK_2, fontsize=12, labelpad=8)
    ax.set_ylabel("val accuracy (%), 500 images", color=INK_2, fontsize=12, labelpad=10)
    fig.text(0.075, 0.945, TITLE, color=INK, fontsize=15, ha="left", va="center")
    ax.text(0.012, 0.03, "chance = 10%", transform=ax.transAxes, color=INK_2,
            fontsize=11, ha="left", va="bottom")
    return fig, ax


class Scene:
    def __init__(self, fig, ax, rows):
        self.fig, self.ax, self.rows = fig, ax, rows
        self.line = Line2D([], [], color=BLUE, lw=2, solid_capstyle="round",
                           solid_joinstyle="round", zorder=3)
        ax.add_line(self.line)
        self.markers = []
        for r in rows:
            m = Line2D([r["i"]], [r["val"]], ls="none", marker="o", ms=9,
                       mfc=RED if r["reverted"] else BLUE, mec=SURFACE, mew=2,
                       zorder=4, visible=False)
            ax.add_line(m)
            self.markers.append(m)
        self.clock = ax.text(0.988, 0.03, "", transform=ax.transAxes, color=INK_2,
                             fontsize=12, ha="right", va="bottom")
        self.callout = None
        self.reverted_label = None
        self.summary = None
        self.placement = {}

    def _clear_callout(self):
        for a in (self.callout, self.reverted_label):
            if a is not None:
                a.remove()
        self.callout = self.reverted_label = None

    def _data_bbox(self, bbox_px):
        inv = self.ax.transData.inverted()
        (x0, y0), (x1, y1) = inv.transform([[bbox_px.x0, bbox_px.y0],
                                            [bbox_px.x1, bbox_px.y1]])
        return Bbox([[x0, y0], [x1, y1]])

    @staticmethod
    def _seg_hits_box(p, q, bb):
        """Liang-Barsky: does segment p->q cross axis-aligned box bb (data units)?"""
        (x0, y0), (x1, y1) = p, q
        dx, dy = x1 - x0, y1 - y0
        t0, t1 = 0.0, 1.0
        for pk, qk in ((-dx, x0 - bb.x0), (dx, bb.x1 - x0), (-dy, y0 - bb.y0), (dy, bb.y1 - y0)):
            if pk == 0:
                if qk < 0:
                    return False
                continue
            t = qk / pk
            if pk < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
            if t0 > t1:
                return False
        return True

    @staticmethod
    def _pt_seg_dist(m, a, b):
        ax_, ay = a; bx, by = b; mx, my = m
        dx, dy = bx - ax_, by - ay
        L2 = dx * dx + dy * dy
        t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((mx - ax_) * dx + (my - ay) * dy) / L2))
        px, py = ax_ + t * dx, ay + t * dy
        return ((mx - px) ** 2 + (my - py) ** 2) ** 0.5

    def _make_callout(self, k, ha, dx, dy):
        r = self.rows[k]
        color = RED if r["reverted"] else BLUE
        return self.ax.annotate(
            f"#{r['i']}: {r['prim']}", xy=(r["i"], r["val"]),
            xytext=(r["i"] + dx, r["val"] + dy), textcoords="data", ha=ha,
            va="top" if dy < 0 else "bottom", fontsize=12,
            color=color if r["reverted"] else INK, zorder=6,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=color, lw=1),
            arrowprops=dict(arrowstyle="-", color=color, lw=1, shrinkA=0, shrinkB=7))

    def _choose_placement(self, k):
        """Try above/below offsets x right/left/centre; keep the first candidate whose
        box is inside the axes and clear of all markers and line segments (including
        the segment that will grow into the next point). Cached per row."""
        if k in self.placement:
            return self.placement[k]
        r = self.rows[k]
        renderer = self.fig.canvas.get_renderer()
        ax_bb = self._data_bbox(self.ax.get_window_extent(renderer))
        pts = [(rr["i"], rr["val"]) for rr in self.rows[: k + 2]]
        segs = list(zip(pts[:-1], pts[1:]))
        above = [1.6, 3.0, 4.6]
        dys = [-d for d in above] + above if r["val"] > BELOW_IF_ABOVE else above + [-d for d in above]
        pad_x, pad_y = 0.2, 0.45  # marker radius plus a little air, in data units
        best, best_pen = None, None
        for dy in dys:
            for ha, dx in (("left", 0.25), ("right", -0.25), ("center", 0.0)):
                ann = self._make_callout(k, ha, dx, dy)
                ann.update_positions(renderer)
                ann.update_bbox_position_size(renderer)
                bb = self._data_bbox(ann.get_bbox_patch().get_window_extent(renderer))
                ann.remove()
                pen = 0.0
                if bb.x0 < ax_bb.x0 or bb.x1 > ax_bb.x1 or bb.y0 < ax_bb.y0 or bb.y1 > ax_bb.y1:
                    pen += 1000.0
                grown = Bbox([[bb.x0 - pad_x, bb.y0 - pad_y], [bb.x1 + pad_x, bb.y1 + pad_y]])
                pen += sum(1.0 for (mx, my) in pts[:k] if grown.x0 <= mx <= grown.x1 and grown.y0 <= my <= grown.y1)
                pen += sum(1.0 for p, q in segs if self._seg_hits_box(p, q, grown))
                # the arrow runs from the box centre to the point: keep it clear of markers
                a = self.ax.transData.transform(((bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2))
                b = self.ax.transData.transform((r["i"], r["val"]))
                for m in self.ax.transData.transform(pts[:k]):
                    if self._pt_seg_dist(m, a, b) < 20:  # px: marker radius + ring + air
                        pen += 1.0
                if best is None or pen < best_pen:
                    best, best_pen = (ha, dx, dy), pen
                if pen == 0.0:
                    break
            if best_pen == 0.0:
                break
        if best_pen > 0:
            print(f"warning: callout #{r['i']} could not avoid all overlaps (penalty {best_pen})")
        self.placement[k] = best
        return best

    def _place_callout(self, k):
        ha, dx, dy = self._choose_placement(k)
        self.callout = self._make_callout(k, ha, dx, dy)
        r = self.rows[k]
        if r["reverted"]:
            self.reverted_label = self.ax.annotate(
                "reverted", xy=(r["i"], r["val"]), xytext=(9, -9),
                textcoords="offset points", ha="left", va="top",
                fontsize=12, fontweight="bold", color=RED, zorder=6)

    def render(self, k, frac, callout_k, minutes, final):
        """k: index of the newest segment end (0-based); frac: growth fraction of
        the segment into point k (1.0 = complete); callout_k: row index whose callout
        is shown (or None)."""
        rows = self.rows
        xs, ys = [], []
        if k is not None:
            xs = [r["i"] for r in rows[: k + 1]]
            ys = [r["val"] for r in rows[: k + 1]]
            if frac < 1.0 and k > 0:
                px, py = rows[k - 1]["i"], rows[k - 1]["val"]
                xs[-1] = px + (rows[k]["i"] - px) * frac
                ys[-1] = py + (rows[k]["val"] - py) * frac
            elif frac < 1.0 and k == 0:
                xs, ys = [], []
        self.line.set_data(xs, ys)
        for j, m in enumerate(self.markers):
            m.set_visible(k is not None and (j < k or (j == k and frac >= 1.0)))
        self.clock.set_text(f"t = {minutes:.1f} min")
        self._clear_callout()
        if callout_k is not None:
            self._place_callout(callout_k)
        if self.summary is not None:
            self.summary.remove()
            self.summary = None
        if final:
            peak = max(rows, key=lambda r: r["val"])
            end = rows[-1]
            self.summary = self.ax.text(
                0.012, 0.965,
                f"peak {peak['val']:.1f}% (it. {peak['i']}) · end {end['val']:.1f}% (it. {end['i']})",
                transform=self.ax.transAxes, color=INK, fontsize=13, ha="left", va="top",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GRID, lw=1), zorder=6)


def timeline(rows, fps):
    """Yield (k, frac, callout_k, minutes, final) per frame."""
    n_intro = round(INTRO_S * fps)
    n_grow = max(1, round(GROW_S * fps))
    n_hold = round(HOLD_S * fps)
    n_final = round(FINAL_S * fps)
    for _ in range(n_intro):
        yield None, 0.0, None, 0.0, False
    for k, r in enumerate(rows):
        prev_min = rows[k - 1]["min"] if k > 0 else 0.0
        for g in range(1, n_grow + 1):
            frac = g / n_grow
            if k == 0:
                frac = 1.0 if g == n_grow else 0.0
            m = prev_min + (r["min"] - prev_min) * (g / n_grow)
            yield k, frac if g < n_grow else 1.0, (k - 1 if k > 0 else None), m, False
        for _ in range(n_hold):
            yield k, 1.0, k, r["min"], False
    last = len(rows) - 1
    for _ in range(n_final):
        yield last, 1.0, last, rows[last]["min"], True


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def encode_gif(frames_pat, fps, out, width, gif_fps):
    vf = (f"fps={gif_fps},scale={width}:-1:flags=lanczos,split[s0][s1];"
          f"[s0]palettegen=max_colors=128:stats_mode=diff[p];"
          f"[s1][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle")
    run(["ffmpeg", "-y", "-v", "error", "-framerate", str(fps), "-i", frames_pat,
         "-vf", vf, "-loop", "0", out])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="run/iterations.csv")
    ap.add_argument("--out", required=True, help="output stem, e.g. media/val_vs_iteration")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--w", type=int, default=1280)
    ap.add_argument("--h", type=int, default=720)
    ap.add_argument("--gif-w", type=int, default=960)
    ap.add_argument("--gif-fps", type=int, default=12)
    ap.add_argument("--frames-dir", default="media/frames")
    ap.add_argument("--no-encode", action="store_true", help="write frames only")
    args = ap.parse_args()

    rows = read_rows(args.csv)
    os.makedirs(args.frames_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    for stale in glob.glob(os.path.join(args.frames_dir, "f[0-9][0-9][0-9][0-9].png")):
        os.remove(stale)

    fig, ax = build_figure(args.w, args.h)
    scene = Scene(fig, ax, rows)
    n = 0
    for n, (k, frac, ck, minutes, final) in enumerate(timeline(rows, args.fps)):
        scene.render(k, frac, ck, minutes, final)
        fig.savefig(os.path.join(args.frames_dir, f"f{n:04d}.png"), dpi=fig.dpi, facecolor=SURFACE)
    n_frames = n + 1
    print(f"wrote {n_frames} frames ({n_frames / args.fps:.2f} s) to {args.frames_dir}")
    if args.no_encode:
        return

    pat = os.path.join(args.frames_dir, "f%04d.png")
    mp4 = args.out + ".mp4"
    gif = args.out + ".gif"
    run(["ffmpeg", "-y", "-v", "error", "-framerate", str(args.fps), "-i", pat,
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
         "-movflags", "+faststart", mp4])
    encode_gif(pat, args.fps, gif, args.gif_w, args.gif_fps)
    size = os.path.getsize(gif)
    if size > 8 * 1024 * 1024:
        print(f"gif is {size / 1e6:.1f} MB > 8 MB; retrying at 800 px / 10 fps")
        encode_gif(pat, args.fps, gif, 800, 10)
        size = os.path.getsize(gif)
    print(f"{mp4}: {os.path.getsize(mp4) / 1e6:.2f} MB")
    print(f"{gif}: {size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
