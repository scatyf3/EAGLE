#!/usr/bin/env python3
"""
Generate a nicely formatted HTML benchmark table.
Rows = context length buckets; column groups = AR / EAGLE / TriForce.
"""

import csv, os

TSV = "outputs/context_runs/llama2_eagle_vs_triforce_buckets.tsv"
OUT = "outputs/context_runs/benchmark_table.html"

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    background: #f4f6f9;
    padding: 36px;
    color: #222;
}
.container {
    max-width: 1400px;
    margin: 0 auto;
    background: #fff;
    border-radius: 10px;
    box-shadow: 0 4px 24px rgba(0,0,0,.10);
    padding: 32px 36px 40px;
}
h2 {
    font-size: 1.1em;
    color: #1a2636;
    font-weight: 700;
    margin-bottom: 6px;
    letter-spacing: .02em;
}
.subtitle {
    font-size: .82em;
    color: #69788a;
    margin-bottom: 22px;
}
table {
    border-collapse: collapse;
    width: 100%;
    font-size: .84em;
}
/* ── header rows ─────────────────────────────── */
thead tr:first-child th {
    padding: 10px 14px 6px;
    border-bottom: none;
    font-size: .93em;
    font-weight: 700;
    letter-spacing: .04em;
    text-transform: uppercase;
    color: #fff;
    text-align: center;
}
thead tr:last-child th {
    padding: 5px 10px 8px;
    font-size: .78em;
    font-weight: 600;
    color: rgba(255,255,255,.85);
    text-align: right;
    border-top: 1px solid rgba(255,255,255,.2);
}
thead tr:first-child th:first-child {
    background: #2d3a4a;
    border-radius: 6px 0 0 0;
}
/* group colours */
th.g-ctx  { background: #2d3a4a; }
th.g-ar   { background: #546e7a; }
th.g-ar-s { background: #607d8b; }
th.g-egl  { background: #1a6b3c; }
th.g-egl-s{ background: #2e7d52; }
th.g-tri  { background: #7b3f00; color: #fff; }
th.g-tri-s{ background: #8d4e00; color: rgba(255,255,255,.85); }

/* speedup columns get a light tint in the body */
td.spd    { background: #f0faf4 !important; font-weight: 600; }
td.spd-bad{ background: #fdf3f0 !important; font-weight: 600; color: #c0392b; }

/* ── body rows ──────────────────────────────── */
tbody tr { border-bottom: 1px solid #ebeef2; }
tbody tr:last-child { border-bottom: 2px solid #c5cdd6; }
tbody tr:hover td { background: #f0f4f8; }
td {
    padding: 8px 10px;
    text-align: right;
    white-space: nowrap;
    color: #2c3e50;
}
td.ctx {
    text-align: center;
    font-weight: 700;
    font-family: "Courier New", monospace;
    font-size: .9em;
    color: #1a2636;
    background: #eceff4 !important;
    border-right: 2px solid #c5cdd6;
}

/* method section separators */
td.sep-ar   { border-left: 2px solid #546e7a44; }
td.sep-egl  { border-left: 2px solid #1a6b3c44; }
td.sep-tri  { border-left: 2px solid #7b3f0044; }

/* highlight best value per column (class added by Python) */
td.best { color: #1a6b3c; font-weight: 700; }

/* footer note */
.note {
    font-size: .77em;
    color: #888;
    margin-top: 16px;
    line-height: 1.6;
}
/* top border under header */
thead tr:last-child th { border-bottom: 2px solid rgba(255,255,255,.35); }
thead + tbody tr:first-child td { border-top: 2px solid #c5cdd6; }
"""

BUCKET_LABELS = {
    "lt256": "<256",
    "256-512": "256–512",
    "512-1k": "512–1k",
    "1k-2k": "1k–2k",
    "2k-4k": "2k–4k",
    "4k-8k": "4k–8k",
}


def read_tsv(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def f(val, d=2):
    try:
        return f"{float(val):.{d}f}"
    except Exception:
        return str(val)


def spd(ar_lat, m_lat):
    try:
        r = float(ar_lat) / float(m_lat)
        return f"{r:.2f}×"
    except Exception:
        return "—"


def spd_val(ar_lat, m_lat):
    try:
        return float(ar_lat) / float(m_lat)
    except Exception:
        return None


def build_table(rows):
    # Pre-compute values for best-column highlighting
    # Columns where higher = better: dec_tps, speedups
    # Columns where lower  = better: lat, prefill
    # We track per-column index → list of float values

    # Column order we'll get per data row (after 'ctx'):
    # AR:       prefill_s, dec_lat_s, dec_tps,  total_lat_s
    # EAGLE:    prefill_s, acc_rate,  dec_tps,  dec_lat_s,  spd_egl
    # TriForce: budget,    prefill_s, s1_acc,   s2_acc,     dec_tps,  dec_lat_s, spd_tri

    # Higher-is-better column indices (0-based, after ctx column):
    higher_better = {2, 7, 9, 13, 18}  # dec_tps columns + speedups
    lower_better  = {0, 1, 3, 5, 8, 11, 12, 14, 15, 17}  # prefill & lat cols

    n_cols = 19    # excluding ctx
    col_vals = [[] for _ in range(n_cols)]          # (float, row_idx)

    data = []
    for r in rows:
        #  AR
        ar_pf  = float(r["ar_prefill_s"])
        ar_dl  = float(r["ar_latency_s"]) - float(r["ar_prefill_s"])
        ar_dt  = float(r["ar_decode_tps"])
        ar_lat = float(r["ar_latency_s"])
        #  EAGLE
        e_pf  = float(r["eagle_prefill_s"])
        e_acc = float(r["eagle_accept_rate"])
        e_dt  = float(r["eagle_decode_tps"])
        e_dl  = float(r["eagle_latency_s"]) - float(r["eagle_prefill_s"])
        e_spd = spd_val(r["ar_latency_s"], r["eagle_latency_s"])
        #  TriForce
        t_bud = int(r["triforce_budget"])
        t_pf  = float(r["triforce_prefill_s"])
        t_s1  = float(r["triforce_s1_rate"])
        t_s2  = float(r["triforce_s2_rate"])
        t_dt  = float(r["triforce_decode_tps"])
        t_dl  = float(r["triforce_latency_s"]) - float(r["triforce_prefill_s"])
        t_spd = spd_val(r["ar_latency_s"], r["triforce_latency_s"])

        vals = [
            ar_pf, ar_dl, ar_dt, ar_lat,          # 0-3
            e_pf, e_acc, e_dt, e_dl, e_spd,       # 4-8
            float(t_bud), t_pf, t_s1, t_s2, t_dt, t_dl, t_spd,  # 9-15
        ]
        data.append((r, vals))

    # Find best per column (compare AR / EAGLE / TriForce on same metric)
    # We'll highlight across all rows per column
    # (Optional: you could highlight within each row instead)
    col_all = [[] for _ in range(16)]
    for _, vals in data:
        for ci, v in enumerate(vals):
            if v is not None:
                col_all[ci].append(v)

    col_best = {}
    for ci in range(16):
        if not col_all[ci]:
            continue
        if ci in {2, 5, 6, 7, 8, 11, 12, 13, 14, 15}:   # higher better (tps, acc, spd)
            col_best[ci] = max(col_all[ci])
        else:                                              # lower better (lat, prefill, budget)
            col_best[ci] = min(col_all[ci])

    # ────────────────────────────────────────────────────────────────────────
    lines = []

    # Header row 1 – group labels
    lines.append('<thead>')
    lines.append('<tr>')
    lines.append('<th class="g-ctx" rowspan="2" style="vertical-align:middle;text-align:center;">Context<br>Length</th>')
    lines.append('<th class="g-ar"  colspan="4">AR — Autoregressive Baseline</th>')
    lines.append('<th class="g-egl" colspan="5">EAGLE</th>')
    lines.append('<th class="g-tri" colspan="7">TriForce</th>')
    lines.append('</tr>')

    # Header row 2 – sub-column labels
    lines.append('<tr>')
    for h in ['Prefill (s)', 'Dec. Lat. (s)', 'Dec. TPS', 'Total Lat. (s)']:
        lines.append(f'<th class="g-ar-s">{h}</th>')
    for h in ['Prefill (s)', 'Acc. Rate', 'Dec. TPS', 'Dec. Lat. (s)', 'Speedup ↑']:
        lines.append(f'<th class="g-egl-s">{h}</th>')
    for h in ['Budget', 'Prefill (s)', 'S1 Acc', 'S2 Acc', 'Dec. TPS', 'Dec. Lat. (s)', 'Speedup ↑']:
        lines.append(f'<th class="g-tri-s">{h}</th>')
    lines.append('</tr>')
    lines.append('</thead>')

    # Body rows
    lines.append('<tbody>')
    for r, vals in data:
        bucket = BUCKET_LABELS.get(r["bucket"], r["bucket"])
        lines.append('<tr>')
        lines.append(f'<td class="ctx">{bucket}</td>')

        def cell(ci, fmt_v, extra_cls=""):
            v = vals[ci]
            best = col_best.get(ci)
            is_best = (v is not None and best is not None and abs(v - best) < 1e-9)
            classes = []
            if extra_cls:
                classes.append(extra_cls)
            # speedup columns: green if > 1x, red if < 1x
            if extra_cls in ("spd", "spd-bad"):
                pass
            elif is_best:
                classes.append("best")
            cls = (' class="' + " ".join(classes) + '"') if classes else ""
            return f'<td{cls}>{fmt_v}</td>'

        def spd_cell(ci, ar_lat, m_lat):
            sv = spd_val(ar_lat, m_lat)
            txt = spd(ar_lat, m_lat)
            if sv is None:
                return '<td>—</td>'
            cls = "spd" if sv >= 1.0 else "spd-bad"
            return f'<td class="{cls}">{txt}</td>'

        ar_lat = r["ar_latency_s"]

        # AR (4 cols)
        lines.append(cell(0, f(vals[0], 3), "sep-ar"))
        lines.append(cell(1, f(vals[1], 3)))
        lines.append(cell(2, f(vals[2], 1)))
        lines.append(cell(3, f(vals[3], 3)))

        # EAGLE (5 cols)
        lines.append(cell(4, f(vals[4], 3), "sep-egl"))
        lines.append(cell(5, f(vals[5], 2)))
        lines.append(cell(6, f(vals[6], 1)))
        lines.append(cell(7, f(vals[7], 3)))
        lines.append(spd_cell(8, ar_lat, r["eagle_latency_s"]))

        # TriForce (7 cols)
        lines.append(cell(9, str(int(vals[9])), "sep-tri"))
        lines.append(cell(10, f(vals[10], 3)))
        lines.append(cell(11, f(vals[11], 3)))
        lines.append(cell(12, f(vals[12], 3)))
        lines.append(cell(13, f(vals[13], 1)))
        lines.append(cell(14, f(vals[14], 3)))
        lines.append(spd_cell(15, ar_lat, r["triforce_latency_s"]))

        lines.append('</tr>')

    lines.append('</tbody>')
    return "\n".join(lines)


def main():
    rows = read_tsv(TSV)
    table_html = build_table(rows)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Speculative Decoding Benchmark — Llama2-7B</title>
  <style>{CSS}</style>
</head>
<body>
<div class="container">
  <h2>Speculative Decoding Benchmark &mdash; Llama2-7B (NousResearch/Yarn-Llama-2-7b-128k)</h2>
  <p class="subtitle">
    gen_len&nbsp;=&nbsp;32 &nbsp;|&nbsp; n_samples&nbsp;=&nbsp;10 &nbsp;|&nbsp;
    temperature&nbsp;=&nbsp;0.6 &nbsp;|&nbsp; top_p&nbsp;=&nbsp;0.9 &nbsp;|&nbsp;
    EAGLE: depth=6, top_k=10, total_token=auto &nbsp;|&nbsp;
    TriForce: gamma=6, budget=ctx/4 &nbsp;|&nbsp; GPU: single A100/RTX
  </p>
  <table>
    {table_html}
  </table>
  <p class="note">
    <strong>Dec. Lat.</strong> = total latency &minus; prefill latency (decode phase only) &nbsp;|&nbsp;
    <strong>Speedup</strong> = AR&nbsp;total&nbsp;latency &divide; method&nbsp;total&nbsp;latency &nbsp;|&nbsp;
    <strong>Acc. Rate</strong> = avg accepted tokens per draft step &nbsp;|&nbsp;
    <strong>S1/S2 Acc</strong> = TriForce stage-1/2 acceptance rates &nbsp;|&nbsp;
    <span style="color:#1a6b3c;font-weight:700;">Bold green</span> = best value in column.
    &nbsp;<span style="color:#c0392b;">Red speedup</span> = slower than AR baseline.
  </p>
</div>
</body>
</html>"""

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fout:
        fout.write(html)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
