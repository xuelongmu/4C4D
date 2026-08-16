#!/usr/bin/env python
"""Build a self-contained local HTML report comparing training experiments.

Scans run directories under an output root for `train.log` (metric
trajectories, wall time, gaussian counts) and `rendered_images/` (held-out
probe renders), and writes one HTML file with every image embedded as a
base64 JPEG so the report can be shared as a single local file.

The generated report embeds capture imagery whose redistribution licensing
has not been established (see docs/experiments/2026-08-08-xuelong-depthkit-rgb.md),
so keep the output file out of Git and off public hosting.

Usage:
  python scripts/build_experiment_report.py \
      --output-root /path/to/output/Xuelong/clip_f300_5s_rgb_posefix \
      --out /path/to/report.html
"""
import argparse
import base64
import html
import json
import os
import re

import cv2

# Ordered manifest: (run_dir, title, phase, verdict, issue)
RUNS = [
    ("ab8-control",        "Control (pre-fix code)",              "1. Tier-0 A/B",   "baseline", ""),
    ("ab8-fixed",          "All bug fixes #5-#12",                "1. Tier-0 A/B",   "regressed: -1.9 dB held-out -> bisected", ""),
    ("ab8-fix7",           "Bisect: #7 only (decay once/step)",   "2. Bisect",       "WIN: +0.6 held-out, +1.2 train — kept", "7"),
    ("ab8-fix5",           "Bisect: #5 only (temporal densify)",  "2. Bisect",       "overfits: train up, held-out -1.4 — reverted", "5"),
    ("ab8-fix10",          "Bisect: #10 only (temporal prune)",   "2. Bisect",       "primary regression: -1.8 held-out — reverted", "10"),
    ("ab8-no5",            "Bundle minus #5 (incl. #10)",         "2. Bisect",       "confirms #10 as main culprit", ""),
    ("ab8-ship",           "Ship tip (reverts applied)",          "3. Ship",         "quality parity, ~30% faster — new baseline", ""),
    ("ab8-shipcache",      "Ship + GPU cache",                    "3. Ship",         "quality-inert, -17%+ wall — kept", "15"),
    ("ab8-budget1m",       "1M gaussian budget",                  "4. Enhancements", "WIN: held-out +0.2, half size, -18% wall — adopted", "18"),
    ("ab8-sqrtlr",         "sqrt-batch LR (x2)",                  "4. Enhancements", "+0.5 once; not replicated — rejected", "16"),
    ("ab8-combo",          "Budget + sqrtLR + cache",             "4. Enhancements", "negative interaction: 19.88 — do not stack", ""),
    ("ab8-coloraffine",    "Per-camera color affine",             "4. Enhancements", "no held-out gain on this rig — flag kept, off", "21"),
    ("ab8-fastprofile",    "Production: budget+cache",            "5. Profiles",     "ADOPTED: 18:29 wall, held-out parity", ""),
    ("ab8-qualityprofile", "Candidate: sqrtLR+cache",             "5. Profiles",     "did not replicate sqrt gain", ""),
    ("ab8-fast-s43",       "Production profile, seed 43",         "6. Replication",  "confirms stability (20.39)", ""),
    ("ab8-sqrtlr-s43",     "sqrtLR+cache, seed 43",               "6. Replication",  "19.42 — confirms sqrt-LR rejection", ""),
    ("ab8-staticfreeze",   "Static-temporal freeze",              "7. Static split", "WIN: +0.73 held-out vs control — adopted", "20"),
    ("ab8-staticfreeze-s43", "Static-temporal freeze, seed 43",   "7. Static split", "WIN replicated: +0.56 held-out", "20"),
]

PROBE = "cam06_0036"
ISSUES_URL = "https://github.com/xuelongmu/4C4D/issues/"

EVAL_RE = re.compile(r"\[ITER (\d+)\] Evaluating (train|test): L1 ([0-9.]+) PSNR ([0-9.]+)")
WALL_RE = re.compile(r"7500/7500 \[(\d+):(\d+)")
GS_RE = re.compile(r"gs_num=(\d+)")


def parse_log(path):
    metrics = {"train": {}, "test": {}}
    wall, gs = None, None
    if not os.path.exists(path):
        return metrics, wall, gs
    text = open(path, errors="replace").read()
    for m in EVAL_RE.finditer(text):
        metrics[m.group(2)][int(m.group(1))] = (float(m.group(3)), float(m.group(4)))
    walls = WALL_RE.findall(text)
    if walls:
        wall = f"{walls[-1][0]}:{walls[-1][1]}"
    gss = GS_RE.findall(text)
    if gss:
        gs = int(gss[-1])
    return metrics, wall, gs


def embed_image(path, width=640, quality=85):
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    if w > width:
        img = cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def probe_render(run_dir):
    for it in (7500, 7000, 6000):
        p = os.path.join(run_dir, "rendered_images", f"test_iter_{it}_cam_{PROBE}.png")
        if os.path.exists(p):
            return p, it
    return None, None


def svg_chart(series, width=960, height=360, y_min=17.0, y_max=21.5):
    """series: list of (label, color, {iter: psnr})"""
    pad_l, pad_b, pad_t = 46, 28, 10
    x_max = 7500
    parts = [f'<svg viewBox="0 0 {width} {height}" style="width:100%;background:#161a20;border-radius:8px">']
    for yv in [17, 18, 19, 20, 21]:
        y = pad_t + (height - pad_b - pad_t) * (1 - (yv - y_min) / (y_max - y_min))
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width-8}" y2="{y:.1f}" stroke="#2a3038" stroke-width="1"/>')
        parts.append(f'<text x="{pad_l-6}" y="{y+4:.1f}" fill="#8a94a2" font-size="12" text-anchor="end">{yv}</text>')
    for xv in [1500, 3000, 4500, 6000, 7500]:
        x = pad_l + (width - pad_l - 8) * xv / x_max
        parts.append(f'<text x="{x:.1f}" y="{height-8}" fill="#8a94a2" font-size="12" text-anchor="middle">{xv}</text>')
    for label, color, pts in series:
        if not pts:
            continue
        coords = []
        for it in sorted(pts):
            x = pad_l + (width - pad_l - 8) * it / x_max
            y = pad_t + (height - pad_b - pad_t) * (1 - (pts[it] - y_min) / (y_max - y_min))
            coords.append(f"{x:.1f},{y:.1f}")
        parts.append(f'<polyline points="{" ".join(coords)}" fill="none" stroke="{color}" stroke-width="2" opacity="0.9"><title>{html.escape(label)}</title></polyline>')
        lx, ly = coords[-1].split(",")
        parts.append(f'<circle cx="{lx}" cy="{ly}" r="3" fill="{color}"><title>{html.escape(label)}</title></circle>')
    parts.append("</svg>")
    return "".join(parts)


PALETTE = ["#e05252", "#e0a052", "#d6d652", "#7ac74f", "#4fc7a0", "#4fa9c7", "#527ae0",
           "#8f52e0", "#c752c7", "#c75283", "#9aa5b1", "#6b7683", "#f0f0f0", "#a0e052",
           "#52e0c7", "#e052a0"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gt_uri = None
    cards = []
    chart_series = []
    rows = []
    for i, (run, title, phase, verdict, issue) in enumerate(RUNS):
        run_dir = os.path.join(args.output_root, run)
        metrics, wall, gs = parse_log(os.path.join(run_dir, "train.log"))
        render_path, render_iter = probe_render(run_dir)
        img_uri = embed_image(render_path) if render_path else None
        if gt_uri is None and render_path:
            gt_path = render_path.replace(".png", "_gt.png")
            if os.path.exists(gt_path):
                gt_uri = embed_image(gt_path)
        test_pts = {it: v[1] for it, v in metrics["test"].items()}
        train_pts = {it: v[1] for it, v in metrics["train"].items()}
        chart_series.append((title, PALETTE[i % len(PALETTE)], test_pts))
        last_test = max(test_pts) if test_pts else None
        last_train = max(train_pts) if train_pts else None
        rows.append((title, phase,
                     f"{train_pts[last_train]:.2f} @{last_train}" if last_train else "—",
                     f"{test_pts[last_test]:.2f} @{last_test}" if last_test else "—",
                     wall or "—", f"{gs:,}" if gs else "—", verdict, issue,
                     PALETTE[i % len(PALETTE)]))
        issue_html = f' · <a href="{ISSUES_URL}{issue}">#{issue}</a>' if issue else ""
        img_html = (f'<img src="{img_uri}" data-render="{img_uri}" loading="lazy" '
                    f'title="hold to compare with ground truth">' if img_uri
                    else "<div class='noimg'>no render</div>")
        cards.append(f"""
<div class="card">
  <div class="cardhead"><span class="dot" style="background:{PALETTE[i % len(PALETTE)]}"></span>
    <b>{html.escape(title)}</b><span class="phase">{html.escape(phase)}{issue_html}</span></div>
  {img_html}
  <div class="meta">held-out {html.escape(rows[-1][3])} · train {html.escape(rows[-1][2])} · wall {html.escape(rows[-1][4])} · {html.escape(rows[-1][5])} gaussians</div>
  <div class="verdict">{html.escape(verdict)}</div>
</div>""")

    table_rows = "".join(
        f"<tr><td><span class='dot' style='background:{c}'></span>{html.escape(t)}</td>"
        f"<td>{html.escape(p)}</td><td>{html.escape(tr)}</td><td><b>{html.escape(te)}</b></td>"
        f"<td>{html.escape(w)}</td><td>{html.escape(g)}</td>"
        f"<td>{html.escape(v)}</td></tr>"
        for t, p, tr, te, w, g, v, _, c in rows)

    # No probe render had a matching _gt.png (empty, partial, or copied output
    # root). Emit valid JS and drop the comparison UI rather than writing
    # `const GT = None;`, which is a ReferenceError that breaks the whole
    # script, and an <img src="None">.
    gt_json = json.dumps(gt_uri)
    if gt_uri:
        gt_section = ('<h2>Ground truth reference (held-out view)</h2>\n'
                      f'<div id="gtbox"><img src="{gt_uri}"></div>')
        gt_hint = ('<b>Press and hold any render to flip to ground truth</b>; '
                   'release to flip back.')
    else:
        gt_section = ('<h2>Ground truth reference (held-out view)</h2>\n'
                      '<div class="noimg">No ground-truth frame found beside the '
                      'probe renders; press-to-compare is disabled.</div>')
        gt_hint = ''

    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>4C4D experiment comparison — Xuelong 8-cam held-out probe</title>
<style>
body {{ background:#0e1116; color:#dde3ea; font:14px/1.5 system-ui, sans-serif; margin:0; padding:24px; }}
h1 {{ font-size:20px; }} h2 {{ font-size:16px; margin-top:32px; }}
a {{ color:#6fb3ff; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(340px,1fr)); gap:16px; }}
.card {{ background:#161a20; border-radius:10px; padding:12px; }}
.card img {{ width:100%; border-radius:6px; cursor:pointer; display:block; }}
.cardhead {{ display:flex; gap:8px; align-items:center; margin-bottom:8px; flex-wrap:wrap; }}
.phase {{ color:#8a94a2; font-size:12px; }}
.meta {{ color:#8a94a2; font-size:12px; margin-top:8px; }}
.verdict {{ font-size:13px; margin-top:4px; }}
.dot {{ display:inline-block; width:10px; height:10px; border-radius:5px; margin-right:6px; }}
table {{ border-collapse:collapse; width:100%; font-size:13px; }}
td, th {{ padding:6px 10px; border-bottom:1px solid #2a3038; text-align:left; }}
.hint {{ color:#8a94a2; }}
.noimg {{ padding:40px; text-align:center; color:#566; background:#10141a; border-radius:6px; }}
#gtbox img {{ max-width:640px; width:100%; border-radius:8px; }}
</style></head><body>
<h1>4C4D experiment comparison</h1>
<p class="hint">Held-out probe: camera <b>cam06</b>, frame 36 (never seen in training).
{gt_hint}
Xuelong scene, 8-camera split, seed 42 unless noted. Full protocol:
<code>docs/experiments/2026-08-10-tier0-bugfixes.md</code> and
<code>2026-08-10-enhancement-experiments.md</code>. Local file — imagery licensing
unestablished, do not redistribute.</p>
{gt_section}
<h2>Held-out PSNR over training</h2>
{svg_chart(chart_series)}
<h2>Experiments</h2>
<div class="grid">{"".join(cards)}</div>
<h2>Summary table</h2>
<table><tr><th>Run</th><th>Phase</th><th>Train PSNR</th><th>Held-out PSNR</th><th>Wall</th><th>Gaussians</th><th>Verdict</th></tr>
{table_rows}</table>
<script>
const GT = {gt_json};
if (GT) {{
  document.querySelectorAll('.card img').forEach(img => {{
    const flip = on => {{ img.src = on ? GT : img.dataset.render; }};
    img.addEventListener('pointerdown', e => {{ e.preventDefault(); flip(true); }});
    ['pointerup','pointerleave','pointercancel'].forEach(ev =>
      img.addEventListener(ev, () => flip(false)));
  }});
}}
</script>
</body></html>"""
    with open(args.out, "w") as f:
        f.write(doc)
    print(f"wrote {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
