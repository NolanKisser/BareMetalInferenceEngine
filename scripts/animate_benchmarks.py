"""Draws the README's animated SVGs from real results. Standard library only.

Usage: python animate_benchmarks.py

Needs benchmarks/results/ and the tinyllama-15M download and export described in
the README. Writes three animations to docs/, each in a light and a dark version:

  anim-trace    One token passing through the engine, before and after
  anim-predict  What the model predicts next, before and after
  anim-race     BareMetal and PyTorch generating the same text, slowed down

They use CSS animation because GitHub READMEs do not run scripts. Viewers who ask
for reduced motion see the finished state.
"""
import array
import json
import pathlib
import re
import struct

from plot_benchmarks import OUTPUT, RESULTS, ROOT, THEMES, escape, read_data, text

MODEL = ROOT / "models" / "tinyllama-15M"
FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"
MONO = "ui-monospace, 'Cascadia Mono', Consolas, Menlo, monospace"
STATUS = {"good": "#0ca30c", "warn": "#fab219", "crit": "#d03b3b"}

# Measured on commit 04397a5, before the engine ran a model. Its benchmark had no
# JSON output and that code is gone, so these are recorded here. Its own output
# said that the rate measured kernel launch overhead, not inference.
OLD_TOKENS_PER_S = 2734
OLD_CHECKS = (10, 13)

# (stage, status before, label before, status after, label after)
STAGES = [
    ("Tokenize text", "warn", "Placeholder", "warn", "Placeholder"),
    ("Load weights", "warn", "Placeholder", "good", "Real"),
    ("Copy weights to VRAM", "good", "Real", "good", "Real"),
    ("KV cache", "warn", "Unused", "good", "Real"),
    ("Embedding lookup", "crit", "Print only", "good", "Real"),
    ("RMSNorm", "crit", "Skipped", "good", "Real"),
    ("RoPE", "crit", "Empty kernel", "good", "Real"),
    ("Attention", "crit", "Missing", "good", "Real"),
    ("Feed-forward", "crit", "Skipped", "good", "Real"),
    ("Final RMSNorm", "crit", "Log only", "good", "Real"),
    ("Output projection", "crit", "Log only", "good", "Real"),
    ("Return logits", "crit", "All zeros", "good", "Real"),
]

# Positions in reference_logits.bin where the model has more than one good answer.
PREDICT = [
    (9, "there was a little"),
    (21, "She loved to play outside in the"),
    (37, "she saw a big, red ball in the"),
]


class Motion:
    """Collects CSS keyframes for one animation loop."""

    def __init__(self, duration, fade_start):
        self.duration = duration
        self.fade = 100 * fade_start / duration
        self.rules = {}

    def pct(self, seconds):
        return 100 * seconds / self.duration

    def _name(self, frames):
        if frames not in self.rules:
            self.rules[frames] = f"a{len(self.rules)}"
        return self.rules[frames]

    def reveal(self, at):
        """Hidden until `at`, then visible until the loop fades out."""
        p = min(self.pct(at), self.fade - 0.02)
        return self._name(f"0%,{p:.2f}%{{opacity:0}}{p + 0.01:.2f}%,{self.fade:.2f}%{{opacity:1}}100%{{opacity:0}}")

    def window(self, start, end, peak=1, edge=0.0):
        """Visible only between `start` and `end`."""
        a, b = self.pct(start), self.pct(end)
        e = self.pct(edge) if edge else 0.01
        before = f"0%,{a:.2f}%{{opacity:0}}" if a > 0 else "0%{opacity:0}"
        return self._name(f"{before}{a + e:.2f}%,{b - e:.2f}%{{opacity:{peak}}}{b:.2f}%,100%{{opacity:0}}")

    def grow(self, start, end):
        """A bar growing from its baseline, then holding."""
        a, b = self.pct(start), self.pct(end)
        return self._name(f"0%,{a:.2f}%{{transform:scaleX(0)}}{b:.2f}%,{self.fade:.2f}%{{transform:scaleX(1)}}100%{{transform:scaleX(1)}}")

    def travel(self, distance, start, end):
        """A marker moving down, then fading out."""
        a, b = self.pct(start), self.pct(end)
        return self._name(f"0%,{a:.2f}%{{transform:translateY(0);opacity:1}}{b:.2f}%{{transform:translateY({distance}px);opacity:1}}"
                          f"{b + 0.5:.2f}%,100%{{transform:translateY({distance}px);opacity:0}}")

    def css(self):
        rules = [f"@keyframes {name}{{{frames}}}.{name}{{animation:{name} {self.duration}s linear infinite}}"
                 for frames, name in self.rules.items()]
        return "\n".join(rules)


def group(css_class, *parts):
    return f'<g class="{css_class}">' + "".join(parts) + "</g>"


def svg(width, height, label, theme, motion, body, extra_css=""):
    c = THEMES[theme]
    style = (f"{motion.css()}\n.grow{{transform-box:fill-box;transform-origin:0 50%}}\n.hide{{opacity:0}}\n{extra_css}\n"
             "@media (prefers-reduced-motion:reduce){*{animation:none!important}}")
    return "\n".join([
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label="{escape(label, {chr(34): "&quot;"})}" font-family="{FONT}">',
        f"<style>{style}</style>",
        f'<rect x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" rx="8" fill="{c["surface"]}" stroke="{c["border"]}"/>',
        *body,
        "</svg>\n",
    ])


def status_mark(x, y, status):
    fill = STATUS[status]
    if status == "good":
        return f'<circle cx="{x}" cy="{y}" r="4.5" fill="{fill}"/>'
    if status == "warn":
        return f'<rect x="{x - 4}" y="{y - 4}" width="8" height="8" rx="1" fill="{fill}" transform="rotate(45 {x} {y})"/>'
    return f'<rect x="{x - 4}" y="{y - 4}" width="8" height="8" rx="1" fill="{fill}"/>'


def draw_trace(theme, checks_passed):
    c = THEMES[theme]
    duration, row_h, top, first, step = 15.0, 28, 142, 0.6, 0.62
    motion = Motion(duration, 14.4)
    width, height = 900, 610
    columns = ((28, "Before", "Commit 04397a5", 1, 2), (472, "After", "Commit a0ad211", 3, 4))
    row_time = lambda i: first + step * i
    done = row_time(len(STAGES))
    bottom = top + row_h * len(STAGES)

    real_after = sum(1 for stage in STAGES if stage[3] == "good")
    real_before = sum(1 for stage in STAGES if stage[1] == "good")
    body = [
        text(28, 46, f"From {real_before} real stage to {real_after}", 23, c["ink"], weight=600),
        text(28, 70, "One token going through the engine. Watch which stages do real work.", 14, c["ink2"]),
    ]
    for x, title, note, status_i, label_i in columns:
        right = x + 400
        body.append(text(x, 106, title, 16, c["ink"], weight=600))
        body.append(text(x + len(title) * 9 + 10, 106, note, 13, c["muted"]))
        body.append(f'<line x1="{x}" x2="{right}" y1="{top - 8}" y2="{top - 8}" stroke="{c["axis"]}"/>')
        for i, stage in enumerate(STAGES):
            y = top + row_h * i
            active = motion.window(row_time(i) - 0.08, row_time(i) + step + 0.1, 1, 0.12)
            body.append(f'<rect class="hide {active}" x="{x}" y="{y - 4}" width="400" height="{row_h - 2}" rx="4" fill="{c["grid"]}"/>')
            body.append(text(x + 26, y + 14, stage[0], 14, c["ink"]))
            shown = motion.reveal(row_time(i) + step * 0.7)
            body.append(group(shown,
                              status_mark(right - 14, y + 9, stage[status_i]),
                              text(right - 26, y + 14, stage[label_i], 13, c["ink2"], "end")))
            body.append(f'<line x1="{x}" x2="{right}" y1="{y + row_h - 5}" y2="{y + row_h - 5}" stroke="{c["grid"]}"/>')
        travel = motion.travel(row_h * (len(STAGES) - 1), first, row_time(len(STAGES) - 1) + step)
        body.append(f'<circle class="hide {travel}" cx="{x + 13}" cy="{top + 9}" r="5" fill="{c["baremetal"]}" stroke="{c["surface"]}" stroke-width="2"/>')

        # The running count of stages that do real work
        y = bottom + 42
        count, changes = 0, [(0.0, 0)]
        for i, stage in enumerate(STAGES):
            if stage[status_i] == "good":
                count += 1
                changes.append((row_time(i) + step * 0.7, count))
        for j, (at, value) in enumerate(changes):
            final = j == len(changes) - 1
            cls = motion.reveal(at) if final else "hide " + motion.window(at, changes[j + 1][0])
            body.append(group(cls, text(x + 44, y, str(value), 34, c["ink"], "end", 600)))
        body.append(text(x + 54, y - 13, f"of {len(STAGES)} stages", 13, c["ink2"]))
        body.append(text(x + 54, y + 3, "do real work", 13, c["ink2"]))

        checks = OLD_CHECKS if title == "Before" else (checks_passed, checks_passed)
        logits = "All zeros" if title == "Before" else "Match Hugging Face"
        results = motion.reveal(done + 0.2)
        body.append(group(results,
                          text(x + 196, y - 13, f"Kernel checks: {checks[0]} of {checks[1]} pass", 13, c["ink2"]),
                          text(x + 196, y + 3, f"Logits: {logits}", 13, c["ink2"])))

    y = bottom + 72
    body.append(f'<line x1="28" x2="{width - 28}" y1="{y}" y2="{y}" stroke="{c["grid"]}"/>')
    body.append(text(28, y + 22, f"The old benchmark reported {OLD_TOKENS_PER_S:,} tokens/s, but its kernels were empty and it computed nothing from a model,", 12, c["muted"]))
    body.append(text(28, y + 38, "so it can't be compared with today's rate. The tokenizer is still a placeholder: BPE isn't implemented, and benchmarks use token IDs.", 12, c["muted"]))
    label = (f"Animation of one token passing through {len(STAGES)} engine stages. Before, {real_before} stage did real work and logits were all zeros. "
             f"After, {real_after} do, logits match Hugging Face, and {checks_passed} of {checks_passed} kernel checks pass. The tokenizer is still a placeholder.")
    return svg(width, height, label, theme, motion, body)


def load_vocabulary():
    data = json.loads((MODEL / "tokenizer.json").read_text(encoding="utf-8"))
    vocabulary = {index: piece for piece, index in data["model"]["vocab"].items()}
    vocabulary.update({token["id"]: token["content"] for token in data.get("added_tokens", [])})
    return vocabulary


def piece_text(piece):
    return piece.replace("▁", " ").replace("<0x0A>", "\n")


def load_predictions(vocabulary):
    raw = (MODEL / "export" / "reference_logits.bin").read_bytes()
    vocab_size, count = struct.unpack("<ii", raw[:8])
    ids = struct.unpack(f"<{count}i", raw[8:8 + 4 * count])
    logits = array.array("f")
    logits.frombytes(raw[8 + 4 * count:])

    results = []
    for position, context in PREDICT:
        row = logits[position * vocab_size:(position + 1) * vocab_size]
        peak = max(row)
        exps = [2.718281828459045 ** (value - peak) for value in row]
        total = sum(exps)
        top = sorted(range(vocab_size), key=lambda i: -exps[i])[:4]
        if top[0] != ids[position + 1]:
            raise SystemExit(f"The most likely token at position {position} is not the one the sequence continues with")
        results.append((context, [(piece_text(vocabulary[i]).strip(), 100 * exps[i] / total) for i in top], vocab_size))
    return results


def draw_predict(theme, predictions, logit_error):
    c = THEMES[theme]
    each, count = 5.0, len(predictions)
    duration = each * count
    motion = Motion(duration, duration - 0.01)
    width, height = 900, 372
    body = [
        text(28, 46, "From no prediction to a real one", 23, c["ink"], weight=600),
        text(28, 70, "The model's most likely next words, as probabilities", 14, c["ink2"]),
    ]
    label_w, x_before, x_after = 76, 28, 472
    row_h, top = 38, 192
    bar_w_before, bar_w_after = 240, 240

    for index, (context, top_words, vocab_size) in enumerate(predictions):
        start = each * index
        end = start + each
        cls = motion.window(start, end, 1, 0.3)
        if index:
            cls = "hide " + cls
        prompt = f"…{context}"
        char_w = 10.2
        parts = [f'<text x="28" y="112" textLength="{len(prompt) * char_w:.1f}" lengthAdjust="spacingAndGlyphs" font-size="17" '
                 f'font-family="{MONO}" fill="{c["ink"]}">{escape(prompt)}</text>']
        word = top_words[0][0]
        chosen = motion.reveal(start + 3.0)
        if index:
            chosen = "hide " + chosen
        parts.append(group(chosen, f'<text x="{28 + (len(prompt) + 1) * char_w:.1f}" y="112" textLength="{len(word) * char_w:.1f}" '
                                   f'lengthAdjust="spacingAndGlyphs" font-size="17" font-weight="700" font-family="{MONO}" '
                                   f'fill="{c["ink"]}">{escape(word)}</text>'))
        parts.append(text(x_before, 152, "Before", 15, c["ink"], weight=600))
        parts.append(text(x_before + 60, 152, "Every word ties, because all logits are 0", 13, c["ink2"]))
        parts.append(text(x_after, 152, "After", 15, c["ink"], weight=600))
        parts.append(text(x_after + 50, 152, f"The model picks “{top_words[0][0]}”", 13, c["ink2"]))
        uniform = 100 / vocab_size
        for row, (word, probability) in enumerate(top_words):
            y = top + row * row_h - 16
            # Before: the same words at the same scale, all at 1 in 32,000
            base = x_before + label_w
            parts.append(text(base - 10, y + 16, word, 14, c["ink"], "end"))
            parts.append(f'<line x1="{base}" x2="{base}" y1="{y - 4}" y2="{y + 26}" stroke="{c["axis"]}"/>')
            parts.append(f'<rect x="{base}" y="{y + 5}" width="{max(1, bar_w_before * uniform / 100):.1f}" height="14" fill="{c["muted"]}"/>')
            parts.append(text(base + 10, y + 16, f"1 in {vocab_size:,}", 13, c["ink2"]))

            # After: real probabilities
            base = x_after + label_w
            length = max(2.0, bar_w_after * probability / 100)
            fill = c["baremetal"] if row == 0 else c["muted"]
            grow = motion.grow(start + 0.5 + row * 0.15, start + 1.4 + row * 0.15)
            parts.append(text(base - 10, y + 16, word, 14, c["ink"], "end", 700 if row == 0 else 400))
            parts.append(f'<line x1="{base}" x2="{base}" y1="{y - 4}" y2="{y + 26}" stroke="{c["axis"]}"/>')
            radius = min(4, length / 2)
            bar = (f'M{base} {y + 2}H{base + length - radius:.1f}Q{base + length:.1f} {y + 2} {base + length:.1f} {y + 2 + radius}'
                   f'V{y + 22 - radius}Q{base + length:.1f} {y + 22} {base + length - radius:.1f} {y + 22}H{base}Z')
            parts.append(f'<path class="grow {grow}" d="{bar}" fill="{fill}"/>')
            value = motion.reveal(start + 1.4 + row * 0.15)
            if index:
                value = "hide " + value
            parts.append(group(value, text(base + length + 8, y + 17, f"{probability:.1f}%" if probability >= 1 else "<1%", 13, c["ink"], weight=600 if row == 0 else 400)))
        body.append(group(cls, *parts))

    y = height - 42
    body.append(f'<line x1="28" x2="{width - 28}" y1="{y}" y2="{y}" stroke="{c["grid"]}"/>')
    body.append(text(28, y + 24, f"tinyllama-15M. Probabilities come from the Hugging Face reference. BareMetal's logits match it to within {logit_error}, too small to draw.", 12, c["muted"]))
    first_context, first_words, _ = predictions[0]
    label = (f"Animation of next-word predictions. Before, all logits were zero, so every word tied. After, the model continues "
             f"“{first_context}” with “{first_words[0][0]}” at {first_words[0][1]:.1f}%.")
    return svg(width, height, label, theme, motion, body)


def wrap_tokens(pieces, columns):
    """Lays tokens out as (line, column, text), keeping each whole word on one line."""
    # Tokens with no leading space continue the previous word, as "." follows "Lily".
    word_lengths, index = {}, 0
    while index < len(pieces):
        end = index + 1
        while end < len(pieces) and "\n" not in pieces[end] and not pieces[end].startswith(" "):
            end += 1
        word_lengths[index] = sum(len(piece.strip(" ")) for piece in pieces[index:end] if "\n" not in piece)
        index = end

    placed, line, column = [], 0, 0
    for index, piece in enumerate(pieces):
        if "\n" in piece:
            line += 1
            column = 0
            placed.append(None)
            continue
        leading = 1 if piece.startswith(" ") and column else 0
        word = piece.strip(" ")
        if index in word_lengths and column + leading + word_lengths[index] > columns:
            line += 1
            column, leading = 0, 0
        column += leading
        placed.append((line, column, word))
        column += len(word)
    return placed, line + 1


def draw_race(theme, d, tokens, vocabulary):
    c = THEMES[theme]
    slowdown = 40
    bm_total = d["bm_ms"] * d["token_count"] / 1000 * slowdown
    pt_total = d["pt_ms"] * d["token_count"] / 1000 * slowdown
    start = 0.5
    duration = start + pt_total + 3.2
    motion = Motion(duration, duration - 0.5)

    pieces = [piece_text(vocabulary[token]) for token in tokens]
    columns, char_w, line_h = 53, 7.25, 20
    placed, lines = wrap_tokens(pieces, columns)
    top = 170
    width, height = 900, top - 14 + (lines - 1) * line_h + 68
    ratio = d["bm_tps"] / d["pt_tps"]

    body = [
        text(28, 46, f"The same story, {ratio:.1f}× as fast", 23, c["ink"], weight=600),
        text(28, 70, f"Both engines generate the same {d['token_count']} tokens. Slowed {slowdown}× so you can watch.", 14, c["ink2"]),
    ]
    lanes = (
        ("BareMetal", 28, d["bm_ms"], bm_total, c["baremetal"]),
        ("PyTorch with transformers", 472, d["pt_ms"], pt_total, c["pytorch"]),
    )
    count = d["token_count"]
    for name, x, ms, total, fill in lanes:
        body.append(text(x, 106, name, 16, c["ink"], weight=600))
        body.append(f'<rect x="{x}" y="118" width="400" height="6" rx="3" fill="{c["grid"]}"/>')
        grow = motion.grow(start, start + total)
        body.append(f'<rect class="grow {grow}" x="{x}" y="118" width="400" height="6" rx="3" fill="{fill}"/>')

        steps = list(range(0, count + 1, 5))
        for j, tokens_done in enumerate(steps):
            at = start + total * tokens_done / count
            label = f"{tokens_done} tokens · {ms * tokens_done:.0f} ms"
            if tokens_done == count:
                body.append(group(motion.reveal(at), text(x + 400, 106, label, 13, c["ink"], "end", 600)))
            else:
                next_at = start + total * steps[j + 1] / count
                body.append(group("hide " + motion.window(at, next_at), text(x + 400, 106, label, 13, c["ink2"], "end")))

        out = []
        for i, item in enumerate(placed):
            if item is None or not item[2]:
                continue
            line, column, word = item
            at = start + total * (i + 1) / count
            length = len(word) * char_w
            out.append(f'<text class="{motion.reveal(at)}" x="{x + column * char_w:.1f}" y="{top - 14 + line * line_h}" '
                       f'textLength="{length:.1f}" lengthAdjust="spacingAndGlyphs" font-size="13" font-family="{MONO}" fill="{c["ink"]}">{escape(word)}</text>')
        body.extend(out)

    y = height - 44
    body.append(f'<line x1="28" x2="{width - 28}" y1="{y}" y2="{y}" stroke="{c["grid"]}"/>')
    body.append(text(28, y + 24, f"Real output from tinyllama-15M. Each engine runs at its median time per token, {d['bm_ms']:.2f} ms and {d['pt_ms']:.2f} ms, on one {d['device']}. Data: benchmarks/results/", 12, c["muted"]))
    label = (f"Animation of both engines generating the same story, slowed {slowdown} times. BareMetal finishes {count} tokens in "
             f"{d['bm_ms'] * count:.0f} milliseconds and PyTorch in {d['pt_ms'] * count:.0f}.")
    return svg(width, height, label, theme, motion, body)


def main():
    data = read_data()
    vocabulary = load_vocabulary()
    tokens = json.loads((RESULTS / "tokens_baremetal_run1.json").read_text())["tokens"]
    predictions = load_predictions(vocabulary)
    kernel_output = (RESULTS / "kernel_tests.txt").read_text()
    checks_passed = len(re.findall(r"^PASS", kernel_output, re.M))
    if re.search(r"^FAIL", kernel_output, re.M) or not checks_passed:
        raise SystemExit("kernel_tests.txt does not show every check passing")

    OUTPUT.mkdir(exist_ok=True)
    for theme in THEMES:
        files = {
            "anim-trace": draw_trace(theme, checks_passed),
            "anim-predict": draw_predict(theme, predictions, data["logit_error"]),
            "anim-race": draw_race(theme, data, tokens, vocabulary),
        }
        for name, content in files.items():
            path = OUTPUT / f"{name}-{theme}.svg"
            path.write_text(content, encoding="utf-8")
            print(f"Wrote {path.relative_to(ROOT)} ({len(content) // 1024} KB)")


if __name__ == "__main__":
    main()
