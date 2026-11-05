"""Draws the README's benchmark chart from benchmarks/results/. Standard library only.

Usage: python plot_benchmarks.py

Writes docs/benchmarks-light.svg and docs/benchmarks-dark.svg. Stops if the two
engines generated different tokens or ran on different GPUs, because the
comparison would then not be valid.
"""
import json
import math
import pathlib
import re
import statistics
import sys
from xml.sax.saxutils import escape

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "benchmarks" / "results"
OUTPUT = ROOT / "docs"
RUNS = (1, 2)

THEMES = {
    "light": {
        "surface": "#fcfcfb", "border": "#e1e0d9", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#6f6d68",
        "grid": "#e1e0d9", "axis": "#c3c2b7", "baremetal": "#2a78d6", "pytorch": "#eb6834",
    },
    "dark": {
        "surface": "#1a1a19", "border": "#383835", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#a3a198",
        "grid": "#2c2c2a", "axis": "#383835", "baremetal": "#3987e5", "pytorch": "#d95926",
    },
}


def load(name):
    return [json.loads((RESULTS / f"{name}_run{run}.json").read_text()) for run in RUNS]


def pooled_median(results, select):
    return statistics.median(sample for result in results for sample in select(result))


def read_data():
    token_bm, token_pt = load("tokens_baremetal"), load("tokens_pytorch")
    gemm_bm, gemm_pt = load("baremetal"), load("pytorch")

    tokens = [result["tokens"] for result in token_bm + token_pt]
    if any(sequence != tokens[0] for sequence in tokens):
        sys.exit("The engines generated different tokens, so their speeds are not comparable.")
    devices = {result["device"] for result in token_bm + token_pt + gemm_bm + gemm_pt}
    if len(devices) != 1:
        sys.exit(f"Results come from different GPUs: {sorted(devices)}")

    count = token_bm[0]["generated_tokens"]
    logit_text = (RESULTS / "logits_test.txt").read_text()
    logit_error = re.search(r"logits at all \d+ positions: worst error ([\d.]+)", logit_text)
    if not logit_error:
        sys.exit("Could not read the largest logit difference from logits_test.txt")

    sizes = [row["n"] for row in gemm_bm[0]["square"]]
    def gflops(results, index):
        n = sizes[index]
        return 2 * n**3 / pooled_median(results, lambda r: r["square"][index]["samples"]) / 1e9

    return {
        "device": devices.pop().replace("NVIDIA GeForce ", ""),
        "token_count": len(tokens[0]),
        "bm_tps": count / pooled_median(token_bm, lambda r: r["samples"]),
        "pt_tps": count / pooled_median(token_pt, lambda r: r["samples"]),
        "bm_ms": 1000 * pooled_median(token_bm, lambda r: r["samples"]) / count,
        "pt_ms": 1000 * pooled_median(token_pt, lambda r: r["samples"]) / count,
        "samples_each": sum(len(r["samples"]) for r in token_bm),
        "logit_error": f"{float(logit_error.group(1)):.5f}",
        "sizes": sizes,
        "bm_gflops": [gflops(gemm_bm, i) for i in range(len(sizes))],
        "pt_gflops": [gflops(gemm_pt, i) for i in range(len(sizes))],
    }


def compact(value):
    if value >= 1000:
        return f"{value / 1000:.1f}K" if value % 1000 else f"{value / 1000:.0f}K"
    return f"{value:.0f}"


def text(x, y, content, size, fill, anchor="start", weight=400):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}">{escape(content)}</text>')


def bar(x0, x1, y, height, fill, radius=4):
    """A horizontal bar: square at the baseline, rounded at the data end."""
    return (f'<path fill="{fill}" d="M{x0:.1f} {y}H{x1 - radius:.1f}Q{x1:.1f} {y} {x1:.1f} {y + radius}'
            f'V{y + height - radius}Q{x1:.1f} {y + height} {x1 - radius:.1f} {y + height}H{x0:.1f}Z"/>')


def speed_panel(d, c):
    x0, plot_width, top, bottom, axis_max = 28, 340, 182, 318, 1000
    sx = lambda v: x0 + plot_width * v / axis_max
    parts = [
        text(x0, 138, "Token generation", 16, c["ink"], weight=600),
        text(x0, 158, f"Tokens per second, {d['token_count']} tokens, one at a time", 13, c["ink2"]),
    ]
    for tick in (0, 250, 500, 750, 1000):
        color = c["axis"] if tick == 0 else c["grid"]
        parts.append(f'<line x1="{sx(tick):.1f}" x2="{sx(tick):.1f}" y1="{top}" y2="{bottom}" stroke="{color}"/>')
        parts.append(text(sx(tick), bottom + 18, f"{tick:,}", 12, c["muted"], "middle"))
    rows = (
        ("BareMetal", d["bm_tps"], d["bm_ms"], c["baremetal"], top + 22),
        ("PyTorch", d["pt_tps"], d["pt_ms"], c["pytorch"], top + 90),
    )
    for label, tps, ms, fill, y in rows:
        parts.append(f'<rect x="{x0 + 1}" y="{y - 22}" width="{plot_width - 2}" height="20" fill="{c["surface"]}"/>')
        parts.append(text(x0 + 6, y - 6, label, 13, c["ink"], weight=500))
        parts.append(bar(x0, sx(tps), y, 24, fill))
        parts.append(text(sx(tps) + 8, y + 17, f"{tps:,.0f}", 15, c["ink"], weight=600))
        parts.append(text(x0 + 76, y - 6, f"{ms:.2f} ms per token", 12, c["ink2"]))

    parts.append(text(x0, 382, f"{d['token_count']} of {d['token_count']}", 24, c["ink"], weight=600))
    parts.append(text(x0, 402, "generated tokens identical", 13, c["ink2"]))
    parts.append(text(x0 + 200, 382, d["logit_error"], 24, c["ink"], weight=600))
    parts.append(text(x0 + 200, 402, "largest logit difference", 13, c["ink2"]))
    return parts


def gemm_panel(d, c):
    left, right, top, bottom = 524, 762, 182, 366
    low, high = math.log10(1000), math.log10(50000)
    sy = lambda v: bottom - (bottom - top) * (math.log10(v) - low) / (high - low)
    sx = lambda i: left + (right - left) * i / (len(d["sizes"]) - 1)
    lead = d["pt_gflops"][-1] / d["bm_gflops"][-1]
    parts = [
        text(476, 138, "Matrix multiply kernel", 16, c["ink"], weight=600),
        text(476, 158, f"GFLOP/s, log scale. PyTorch's cuBLAS is {lead:.1f}× faster at {d['sizes'][-1]:,}", 13, c["ink2"]),
    ]
    for tick in (1000, 2000, 5000, 10000, 20000, 50000):
        color = c["axis"] if tick == 1000 else c["grid"]
        parts.append(f'<line x1="{left}" x2="{right}" y1="{sy(tick):.1f}" y2="{sy(tick):.1f}" stroke="{color}"/>')
        parts.append(text(left - 8, sy(tick) + 4, compact(tick), 12, c["muted"], "end"))
    for i, n in enumerate(d["sizes"]):
        parts.append(text(sx(i), bottom + 18, f"{n:,}", 12, c["muted"], "middle"))
    parts.append(text((left + right) / 2, bottom + 36, "Matrix size n", 12, c["muted"], "middle"))

    series = (("PyTorch", d["pt_gflops"], c["pytorch"]), ("BareMetal", d["bm_gflops"], c["baremetal"]))
    for _, values, fill in series:
        points = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(values))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{fill}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>')
    for _, values, fill in series:
        for i, v in enumerate(values):
            parts.append(f'<circle cx="{sx(i):.1f}" cy="{sy(v):.1f}" r="4" fill="{fill}" stroke="{c["surface"]}" stroke-width="2"/>')
    for label, values, _ in series:
        y = sy(values[-1])
        parts.append(text(right + 10, y, compact(round(values[-1], -2)), 14, c["ink"], weight=600))
        parts.append(text(right + 10, y + 16, label, 12, c["ink2"]))
    return parts


def alt_text(d):
    return (f"On tinyllama-15M, BareMetal generates {d['bm_tps']:,.0f} tokens per second and PyTorch {d['pt_tps']:,.0f}, "
            f"with identical tokens. On the matrix multiply kernel, PyTorch reaches {d['pt_gflops'][-1]:,.0f} GFLOP/s "
            f"and BareMetal {d['bm_gflops'][-1]:,.0f} at size {d['sizes'][-1]:,}.")


def draw(d, theme):
    c = THEMES[theme]
    width, height = 900, 476
    ratio = d["bm_tps"] / d["pt_tps"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label="{escape(alt_text(d), {chr(34): "&quot;"})}" font-family="system-ui, -apple-system, \'Segoe UI\', Roboto, sans-serif">',
        f'<rect x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" rx="8" fill="{c["surface"]}" stroke="{c["border"]}"/>',
        text(28, 46, f"{ratio:.1f}× the token rate of PyTorch on a small Llama model", 23, c["ink"], weight=600),
        text(28, 70, f"tinyllama-15M, greedy generation on one {d['device']}, median of {d['samples_each']} runs per engine", 14, c["ink2"]),
    ]
    legend_y = 100
    parts.append(f'<rect x="28" y="{legend_y - 6}" width="16" height="4" rx="2" fill="{c["baremetal"]}"/>')
    parts.append(text(50, legend_y, "BareMetal", 13, c["ink2"]))
    parts.append(f'<rect x="130" y="{legend_y - 6}" width="16" height="4" rx="2" fill="{c["pytorch"]}"/>')
    parts.append(text(152, legend_y, "PyTorch with Hugging Face transformers", 13, c["ink2"]))
    parts.extend(speed_panel(d, c))
    parts.extend(gemm_panel(d, c))
    parts.append(f'<line x1="28" x2="{width - 28}" y1="426" y2="426" stroke="{c["grid"]}"/>')
    parts.append(text(28, 448, "One small model only. Larger models spend more of each token in matrix multiplies, where PyTorch is faster,", 12, c["muted"]))
    parts.append(text(28, 464, "so this lead may shrink or reverse. Data: benchmarks/results/", 12, c["muted"]))
    parts.append("</svg>\n")
    return "\n".join(parts)


def main():
    data = read_data()
    OUTPUT.mkdir(exist_ok=True)
    for theme in THEMES:
        path = OUTPUT / f"benchmarks-{theme}.svg"
        path.write_text(draw(data, theme), encoding="utf-8")
        print(f"Wrote {path.relative_to(ROOT)}")
    print(f"Tokens/s: BareMetal {data['bm_tps']:.1f} ({data['bm_ms']:.3f} ms), PyTorch {data['pt_tps']:.1f} ({data['pt_ms']:.3f} ms), "
          f"ratio {data['bm_tps'] / data['pt_tps']:.2f}, samples each {data['samples_each']}")
    for n, bm, pt in zip(data["sizes"], data["bm_gflops"], data["pt_gflops"]):
        print(f"n={n}: BareMetal {bm:,.0f} GFLOP/s, PyTorch {pt:,.0f} GFLOP/s")


if __name__ == "__main__":
    main()
