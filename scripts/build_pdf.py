"""Build a results PDF for a PDR experiment run.

Usage:
    python3 build_pdf.py <run_dir> <seed> <pdf_out_path>

Parses run.log + degree.log for the key statistics, embeds every PNG in
run_dir, and writes a cover page with all numbers + execution timings.
"""
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import fitz  # pymupdf


def parse_log(text: str) -> dict:
    """Extract the numbers we want from a run log."""
    out = {}

    m = re.search(r"(\d+) nodes \| mean degree ([\d.]+) \(min (\d+) / max (\d+)\) \| density ([\d.]+) nodes/cell \(sd [\d.]+, CV ([\d.]+)\)", text)
    if m:
        out["topology"] = {
            "nodes": int(m.group(1)), "avg_deg": float(m.group(2)),
            "min_deg": int(m.group(3)), "max_deg": int(m.group(4)),
            "density": float(m.group(5)), "cv": float(m.group(6)),
        }

    m = re.search(r"Reachability sweep: (\d+) start nodes, (\d+) speeds, (\d+) gravity directions", text)
    if m:
        out["sweep"] = {"starts": int(m.group(1)), "speeds": int(m.group(2)), "dirs": int(m.group(3))}

    m = re.search(r"Sweeping with (\d+) worker processes", text)
    if m:
        out["workers"] = int(m.group(1))

    m = re.search(r"Overall reachability: (\d+)/(\d+) \(([\d.]+)%\)", text)
    if m:
        out["overall"] = {"reach": int(m.group(1)), "total": int(m.group(2)), "pct": float(m.group(3))}

    regions = re.findall(r"Region (\d+): (\d+) possible, (\d+) reachable, ([\d.]+)%", text)
    if regions:
        out["regions"] = [(int(r[0]), int(r[1]), int(r[2]), float(r[3])) for r in regions]

    m = re.search(r"Dijkstra all-pairs: (\d+) pairs \| mean path ([\d.]+) hops \| max (\d+) hops \| computed in ([\d.]+) s", text)
    if m:
        out["dijkstra"] = {"pairs": int(m.group(1)), "mean": float(m.group(2)),
                           "max": int(m.group(3)), "t": float(m.group(4))}

    m = re.search(r"Trajectory search demo from node (\S+) \(angle ([\d.]+) rad, speed_x in \{([^}]+)\}\)", text)
    if m:
        out["traj_demo"] = {"node": m.group(1), "angle": m.group(2), "speeds": m.group(3)}

    for ln in text.splitlines():
        if "Reverse trajectory demo" in ln:
            profiles = re.findall(r"\(([\d.]+ rad, vx [-\d]+, vy [-\d]+)\)", ln)
            mnode = re.search(r"from node (\S+):", ln)
            if mnode:
                out["reverse"] = {"node": mnode.group(1),
                                  "p1": profiles[0] if profiles else "",
                                  "p2": profiles[1] if len(profiles) > 1 else ""}
            break

    m = re.search(r"Execution time: ([\d.]+) s \(wall(?:, (\d+) workers)?\)", text)
    if m:
        out["exec_total"] = float(m.group(1))

    saved = re.findall(r"Saved plot: (\S+)", text)
    if saved:
        out["saved_plots"] = saved

    return out


def parse_degree_log(text: str) -> dict:
    out = {}
    m = re.search(r"Degree experiment results: (\[.*\])", text)
    if m:
        import ast
        out["results"] = ast.literal_eval(m.group(1))
    m = re.search(r"Execution time: ([\d.]+) s \(wall\)", text)
    if m:
        out["exec_total"] = float(m.group(1))
    return out


def fmt_dur(s: float) -> str:
    if s < 90:
        return f"{s:.1f} s"
    m, sec = divmod(s, 60)
    return f"{int(m)} min {sec:04.1f} s"


def collect_pngs(run_dir: Path, region_pct: float | None = None) -> list:
    """Return an ordered list of (caption, path) for every plot produced."""
    if region_pct is not None:
        hm_cap = (f"Regional reachability heat map — 16 regions (2.5 × 2.5 units); "
                  f"region-average {region_pct:.2f}% reachable.")
    else:
        hm_cap = "Regional reachability heat map — 16 regions (2.5 × 2.5 units)."
    order = [
        ("Regional_Reachability_Heat_Map.png", hm_cap),
        ("Trajectory_Search_from_Node_0.png",
         "Trajectory search from node 0 — distinct launch-speed trajectories under gravity "
         "(paper Sec. 10.1: exponential + binary search)."),
        ("Reverse_Trajectory_-_Forward.png",
         "Reverse-trajectory demo, forward phase — two launch profiles traced from node 0 to their first common node."),
        ("Reverse_Trajectory_-_Backwards.png",
         "Reverse-trajectory demo, reverse phase — both trajectories traced back from the common node to the source."),
        ("Dijkstra_All-Pairs_Path-Length_Distribution.png",
         "Dijkstra all-pairs path-length distribution (hop count, 39,800 pairs)."),
        ("Reachability_by_Degree_and_Gravity_Directions.png",
         "Reachability by average degree and mPDR gravity-direction count (paper Fig. 9 style; fast scale, 200 nodes)."),
    ]
    out = []
    for name, caption in order:
        p = run_dir / name
        if p.exists():
            out.append((caption, p))
    known = {n for n, _ in order}
    for p in sorted(run_dir.glob("*.png")):
        if p.name not in known:
            out.append((p.stem.replace("_", " "), p))
    return out


def find_log(run_dir: Path, name: str) -> Path | None:
    p = run_dir / name
    return p if p.exists() else None


def build_pdf(run_dir: Path, seed: int, pdf_path: Path) -> None:
    run_log = find_log(run_dir, "run.log")
    text = run_log.read_text(errors="replace") if run_log else ""
    stats = parse_log(text)

    region_avg = None
    regs = stats.get("regions", [])
    if regs:
        region_avg = sum(r[2] / r[1] * 100 for r in regs) / len(regs)

    degree = {}
    deg_log = find_log(run_dir, "degree.log")
    if deg_log:
        degree = parse_degree_log(deg_log.read_text(errors="replace"))

    plots = collect_pngs(run_dir, region_avg)

    doc = fitz.open()
    page_w, page_h = 595.0, 842.0  # A4
    MARGIN = 50
    MAX_W = page_w - 2 * MARGIN

    p = doc.new_page(width=page_w, height=page_h)
    y = 55
    state = {"y": y}

    def line(txt, size=10.5, bold=False, color=(0, 0, 0), indent=0):
        # wrap manually at ~MAX_W using the font width
        fontname = "HeBo" if bold else "Helv"
        words = txt.split(" ")
        cur = ""
        for w in words:
            trial = (cur + " " + w).strip()
            if fitz.get_text_length(trial, fontname=fontname, fontsize=size) > MAX_W - indent and cur:
                p.insert_text((MARGIN + indent, state["y"]), cur, fontsize=size, fontname=fontname, color=color)
                state["y"] += size * 1.45 + 2
                cur = w
            else:
                cur = trial
        if cur:
            p.insert_text((MARGIN + indent, state["y"]), cur, fontsize=size, fontname=fontname, color=color)
            state["y"] += size * 1.45 + 2

    def gap(px=8):
        state["y"] += px

    # ---------- title ----------
    line("Particle Dynamics Routing — Full Experiment Report", 17, bold=True)
    line(f"200-node planar topology   |   random seed {seed}   |   scale: full (paper parameters)", 11, bold=True, color=(0.15, 0.15, 0.15))
    line(f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}   |   12-worker parallel reachability sweep", 9, color=(0.35, 0.35, 0.35))
    gap(10)

    # ---------- topology ----------
    t = stats.get("topology", {})
    line("1.  Topology", 13, bold=True, color=(0.05, 0.25, 0.45))
    if t:
        line(f"Nodes: {t['nodes']}     Mean degree: {t['avg_deg']:.2f} (min {t['min_deg']} / max {t['max_deg']})     "
             f"Density: {t['density']:.3f} nodes/cell (CV {t['cv']:.3f})", 10.5, indent=12)
        line("10 × 10 unit plane, 10 × 10 cells, connection radius 2.0, min-degree 2 / avg-degree 5. "
             "Crossing-free k-NN links with min/avg-degree correction (planar).", 9.5, indent=12, color=(0.3, 0.3, 0.3))
    gap()

    # ---------- reachability ----------
    o = stats.get("overall", {})
    sw = stats.get("sweep", {})
    line("2.  Reachability (particle routing)", 13, bold=True, color=(0.05, 0.25, 0.45))
    if o and sw:
        line(f"Sweep: {sw['starts']} start nodes × {sw['speeds']} launch speeds × {sw['dirs']} gravity directions = {o['total']:,} trials", 10.5, indent=12)
        line(f"Overall reachability:  {o['reach']:,} / {o['total']:,}  =  {o['pct']:.2f} %", 12, bold=True, indent=12, color=(0.1, 0.5, 0.1))
    regs = stats.get("regions", [])
    if regs:
        lo = min(r[3] for r in regs)
        hi = max(r[3] for r in regs)
        avg_r = sum(r[2] / r[1] * 100 for r in regs) / len(regs)
        line(f"16 regions:  {lo:.1f} % (worst)  →  {hi:.1f} % (best)  |  region-average {avg_r:.2f} %", 10.5, indent=12)
    gap()

    # ---------- dijkstra ----------
    d = stats.get("dijkstra", {})
    line("3.  Dijkstra all-pairs (reference routing)", 13, bold=True, color=(0.05, 0.25, 0.45))
    if d:
        line(f"Pairs: {d['pairs']:,}     Mean shortest path: {d['mean']:.2f} hops     Max: {d['max']} hops     "
             f"Computation: {d['t']:.3f} s", 10.5, indent=12)
    gap()

    # ---------- trajectory demos ----------
    line("4.  Trajectory & reverse-trajectory demos", 13, bold=True, color=(0.05, 0.25, 0.45))
    td = stats.get("traj_demo", {})
    rv = stats.get("reverse", {})
    if td:
        line(f"Forward search from node {td['node']}: angle {td['angle']} rad, speed_x in {{{td['speeds']}}} -> "
             f"distinct trajectories enumerated by binary search (see plot).", 10.5, indent=12)
    if rv:
        line(f"Reverse demo from node {rv['node']}: launch profiles {rv['p1']} and {rv['p2']} → "
             f"forward to common node, then reverse walk back to source.", 10.5, indent=12)
    gap()

    # ---------- degree ----------
    if degree.get("results"):
        line("5.  Reachability vs. average degree (mPDR, paper Fig. 9 style)", 13, bold=True, color=(0.05, 0.25, 0.45))
        line("Degree experiment topology: 200 nodes, fast scale, 11 speeds, 8 gravity directions, 50% source sweep.", 9.5, indent=12, color=(0.3, 0.3, 0.3))
        for avg_deg, dirs in degree["results"]:
            ds = " / ".join(f"{v:.1f}%" for v in dirs)
            line(f"avg degree {avg_deg:.2f}:   1 dir {dirs[0]:5.1f}%   2 dirs {dirs[1]:5.1f}%   4 dirs {dirs[2]:5.1f}%   8 dirs {dirs[3]:5.1f}%", 10.5, indent=12)
        gap()

    # ---------- timings ----------
    line("Execution time (wall clock)", 13, bold=True, color=(0.05, 0.25, 0.45))
    if "exec_total" in stats:
        line(f"Full experiment (this run, seed {seed}):  {fmt_dur(stats['exec_total'])}   — includes topology build, "
             f"{sw.get('starts', 200)}-node parallel reachability sweep (12 workers), heat map, Dijkstra, trajectory demos", 10.5, indent=12)
        line(f"Dijkstra all-pairs alone:  {d.get('t', 0):.3f} s", 10.5, indent=12)
    if degree.get("exec_total"):
        line(f"Degree experiment (fast scale, 4 workers):  {fmt_dur(degree['exec_total'])}", 10.5, indent=12)
    line("Hardware: 24-core host, Python 3.11, ProcessPoolExecutor.", 9.5, indent=12, color=(0.35, 0.35, 0.35))
    gap(12)

    # ---------- plot list ----------
    line("Plots (one per following page)", 13, bold=True, color=(0.05, 0.25, 0.45))
    for i, (caption, _) in enumerate(plots, 1):
        first_line = caption.split(" — ")[0]
        line(f"{i}.  {first_line}", 10, indent=12, color=(0.25, 0.25, 0.25))

    # ---------- one page per plot ----------
    for idx, (caption, png) in enumerate(plots, 1):
        pg = doc.new_page(width=page_w, height=page_h)
        head = caption.split(" — ")[0]
        pg.insert_text((MARGIN, 38), f"{idx}.  {head}", fontsize=12, fontname="HeBo")
        rest = caption.split(" — ", 1)[1] if " — " in caption else ""
        if rest:
            # simple wrap
            words, cur = rest.split(" "), ""
            yy = 56
            for w in words:
                trial = (cur + " " + w).strip()
                if fitz.get_text_length(trial, fontname="Helv", fontsize=9.5) > MAX_W and cur:
                    pg.insert_text((MARGIN, yy), cur, fontsize=9.5, fontname="Helv", color=(0.3, 0.3, 0.3))
                    yy += 14
                    cur = w
                else:
                    cur = trial
            if cur:
                pg.insert_text((MARGIN, yy), cur, fontsize=9.5, fontname="Helv", color=(0.3, 0.3, 0.3))
            img_top = max(62, yy + 14)
        else:
            img_top = 62
        img = fitz.Pixmap(str(png))
        if img.n > 4:
            img = fitz.Pixmap(fitz.csRGB, img)
        box = fitz.Rect(MARGIN, img_top, page_w - MARGIN, page_h - 60)
        scale = min(box.width / img.width, box.height / img.height)
        new_w, new_h = img.width * scale, img.height * scale
        new_r = fitz.Rect(box.x0 + (box.width - new_w) / 2, box.y0,
                          box.x0 + (box.width - new_w) / 2 + new_w, box.y0 + new_h)
        pg.insert_image(new_r, pixmap=img)

    doc.set_metadata({
        "title": f"PDR full experiment — 200 nodes, seed {seed}",
        "author": "particle-dynamics-routing (yyd330)",
        "subject": ("Paper-scale PDR experiment (Biswas et al., PLOS ONE e0357202): "
                    "planar topology, reachability sweep, heat map, trajectory demos, Dijkstra, degree study"),
    })
    pages = doc.page_count
    doc.save(str(pdf_path), garbage=4, deflate=True)
    doc.close()
    print(f"PDF written: {pdf_path} ({os.path.getsize(pdf_path):,} bytes, {pages} pages)")


if __name__ == "__main__":
    run_dir = Path(sys.argv[1])
    seed = int(sys.argv[2])
    pdf_path = Path(sys.argv[3])
    build_pdf(run_dir, seed, pdf_path)
