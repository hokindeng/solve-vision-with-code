#!/usr/bin/env python3
"""Advisory prose lint for paper/sec/*.tex: patterns that read as machine-written.

    python3 paper/lint_prose.py             # run from the repo root; exit 0 (advisory)
    python3 paper/lint_prose.py --strict    # exit 1 when any file scores below --min (default 70)
    python3 paper/lint_prose.py --min 80 --top 30 paper/sec/4_results.tex

Per file, with line numbers and counts:
    dash        paragraph with more than one em dash (--- or the character)
    notbut      "not X but Y", "not only ... but"
    isnt-its    "isn't X, it's Y" and variants
    signpost    sentence starting Crucially / Notably / Importantly / Interestingly / In other words /
                Put differently / The story / The lesson / In short / Ultimately / At its core
    triad       "A, B, and C" with three single words (heuristic, report only)
    summary     paragraph whose last sentence has <= 8 words with this/that/it as subject (summary tell)
    boldlead    paragraph opening with \\textbf{...} (count per section)
    colon       ": " followed by a lowercase clause of <= 6 words (colon-led reveal)
    question    rhetorical question
    we          "we" density per 100 words (penalised above 2.5)
    monotone    standard deviation of sentence length below 5 words

Score per file: 100 - sum over kinds of weight x hits x (1000 / words) - we_penalty - monotone_penalty,
floored at 0 (100 = clean). The weights are printed in the report. Standard library only.
"""
import argparse
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEC = ROOT / "paper" / "sec"

WEIGHTS = {"dash": 3, "notbut": 4, "isnt-its": 4, "signpost": 3, "triad": 1, "summary": 2, "boldlead": 0.5, "colon": 2, "question": 3}
WE_FREE = 2.5          # "we" per 100 words allowed before a penalty
WE_WEIGHT = 4          # penalty per extra "we" per 100 words
MONOTONE_STDEV = 5.0   # words
MONOTONE_PENALTY = 10

SIGNPOSTS = r"(?:Crucially|Notably|Importantly|Interestingly|In other words|Put differently|The story|The lesson|In short|Ultimately|At its core)"
ABBREV = r"(?:e\.g|i\.e|cf|vs|et al|Fig|Eq|Tab|Sec|approx|resp|No|St|Dr|Mr|Ms|Prof)"


# ---------------------------------------------------------------- LaTeX -> text
def strip_tex(line):
    """Comment and markup removal for one source line; the prose (incl. captions) survives."""
    s = re.sub(r"(?<!\\)%.*", "", line)
    if re.match(r"\s*\\(?:begin|end)\{(?:figure\*?|table\*?|minipage|tabular|longtable|itemize|enumerate|abstract|center)\}", s):
        return ""
    s = re.sub(r"\\(?:input|label|includegraphics|verbatiminput|graphicspath|setlength|makebox|centering|makeatletter|makeatother|def\\verbatim@font|small|scriptsize|footnotesize|onecolumn|twocolumn|toprule|midrule|bottomrule|hfill|noindent|item)\b(?:\[[^\]]*\])?(?:\{[^{}]*\})*", " ", s)
    s = re.sub(r"\\(?:cite[pt]?|citealp|cref|Cref|ref|eqref|url|href)\*?(?:\[[^\]]*\])*\{[^{}]*\}", "REF", s)
    s = re.sub(r"\\(?:section|subsection|subsubsection|paragraph|caption)\*?\{", "{", s)
    for _ in range(4):  # nested formatting commands: keep the argument
        s = re.sub(r"\\(?:textbf|emph|textit|texttt|underline|textsc|mbox|text)\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\$[^$]*\$", "MATH", s)
    s = s.replace("\\,", " ").replace("~", " ").replace("\\%", "%").replace("\\_", "_").replace("\\&", "&").replace("\\\\", " ")
    s = re.sub(r"\\[a-zA-Z@]+\*?", " ", s)  # any remaining command name
    s = re.sub(r"[{}]", "", s)
    s = s.replace("``", '"').replace("''", '"')
    return re.sub(r"[ \t]+", " ", s).strip()


def paragraphs(path):
    """-> [(first_line_no, raw_first_line, text, section)] with blank-line paragraph breaks."""
    out, buf, start, raw_first = [], [], None, ""
    section = "(preamble)"
    lines = path.read_text().splitlines()
    for i, raw in enumerate(lines, 1):
        m = re.match(r"\s*\\(?:section|subsection|subsubsection)\*?\{([^{}]*)\}", raw)
        if m:  # a heading is not prose; it ends the paragraph before it
            section = m.group(1)
            text = ""
        else:
            text = strip_tex(raw)
        if not text:
            if buf:
                out.append((start, raw_first, " ".join(buf), section))
                buf = []
            continue
        if not buf:
            start, raw_first = i, raw
        buf.append(text)
    if buf:
        out.append((start, raw_first, " ".join(buf), section))
    return out


def sentences(text):
    protected = re.sub(rf"\b{ABBREV}\.", lambda m: m.group(0).replace(".", "§"), text)
    parts = re.split(r"(?<=[.!?])[\"')]?\s+(?=[A-Z\"(])", protected)
    return [p.replace("§", ".").strip() for p in parts if p.strip()]


def words(s):
    return re.findall(r"[A-Za-z][A-Za-z'-]*", s)


# ---------------------------------------------------------------- checks
def lint_file(path):
    hits = []  # (kind, line, snippet)
    bold_per_section = Counter()
    all_sentence_lengths = []
    n_words = 0
    n_we = 0
    for line_no, raw_first, text, section in paragraphs(path):
        ws = words(text)
        n_words += len(ws)
        n_we += sum(1 for w in ws if w.lower() == "we")
        dashes = text.count("---") + text.count("—")
        if dashes > 1:
            hits.append(("dash", line_no, f"{dashes} em dashes in one paragraph"))
        if raw_first.lstrip().startswith("\\textbf{") and "&" not in raw_first:
            bold_per_section[section] += 1
            hits.append(("boldlead", line_no, raw_first.strip()[:90]))
        sents = sentences(text)
        all_sentence_lengths += [len(words(s)) for s in sents]
        for s in sents:
            if re.search(r"\bnot\b(?! only)[^.;:]{0,80}?,?\s\bbut\b(?! (?:also|not))", s, re.I) or re.search(r"\bnot only\b.{0,120}?\bbut\b", s, re.I):
                hits.append(("notbut", line_no, s))
            if re.search(r"\b(?:isn't|is not|aren't|are not|wasn't|was not)\b[^.;]{0,80}?[,;]\s*(?:it's|it is|they're|they are|it was)\b", s, re.I):
                hits.append(("isnt-its", line_no, s))
            if re.match(rf"^[\"(]?{SIGNPOSTS}\b", s):
                hits.append(("signpost", line_no, s))
            for m in re.finditer(r"\b([A-Za-z-]+), ([A-Za-z-]+),? and ([A-Za-z-]+)(?=[\s.,;:)])", s):
                if all(w.lower() not in ("the", "a", "an", "of", "in", "on", "to") for w in m.groups()):
                    hits.append(("triad", line_no, m.group(0)))
            for m in re.finditer(r":\s+([a-z][A-Za-z'-]*(?:\s+[A-Za-z0-9'-]+){0,5})[.;](?!\d)", s):
                hits.append(("colon", line_no, ": " + m.group(1)))
            if s.rstrip("\"')").endswith("?"):
                hits.append(("question", line_no, s))
        if sents:
            last = sents[-1]
            lw = words(last)
            if len(lw) <= 8 and (re.match(r"^[\"(]?(?:This|That|It|These|Those)\b", last) or re.search(r"\b(?:this|that|it)\s+(?:is|was|does|did|matters|holds|remains|means|shows|counts)\b", last, re.I)):
                hits.append(("summary", line_no, last))
    counts = Counter(k for k, _, _ in hits)
    per_k = 1000 / max(n_words, 1)
    penalty = sum(WEIGHTS[k] * c * per_k for k, c in counts.items())
    we_density = 100 * n_we / max(n_words, 1)
    we_pen = max(0.0, we_density - WE_FREE) * WE_WEIGHT
    mean_len = statistics.mean(all_sentence_lengths) if all_sentence_lengths else 0.0
    stdev_len = statistics.pstdev(all_sentence_lengths) if len(all_sentence_lengths) > 1 else 0.0
    monotone = bool(all_sentence_lengths) and stdev_len < MONOTONE_STDEV
    score = max(0.0, 100 - penalty - we_pen - (MONOTONE_PENALTY if monotone else 0))
    return dict(path=path, hits=hits, counts=counts, words=n_words, we=n_we, we_density=we_density, we_pen=we_pen,
                mean_len=mean_len, stdev_len=stdev_len, monotone=monotone, score=score, penalty=penalty,
                bold_per_section=bold_per_section, sentences=len(all_sentence_lengths))


# ---------------------------------------------------------------- report
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="*", help="default: paper/sec/*.tex")
    ap.add_argument("--strict", action="store_true", help="exit 1 when any file scores below --min")
    ap.add_argument("--min", type=float, default=70.0, help="minimum score for --strict (default 70)")
    ap.add_argument("--top", type=int, default=15, help="how many offending lines to print verbatim (default 15)")
    a = ap.parse_args()
    files = [Path(f) for f in a.files] or sorted(SEC.glob("*.tex"))
    print("== prose lint (advisory) ==")
    print("score = 100 - sum_k weight_k x hits_k x (1000 / words) - max(0, we per 100 words - %.1f) x %d - %d if sentence-length stdev < %.0f words"
          % (WE_FREE, WE_WEIGHT, MONOTONE_PENALTY, MONOTONE_STDEV))
    print("weights: " + ", ".join(f"{k}={v}" for k, v in WEIGHTS.items()))
    print()
    kinds = list(WEIGHTS)
    head = f"{'file':22s} {'score':>5s} {'words':>5s} {'sent':>4s} {'mean':>5s} {'stdev':>5s} {'we/100':>6s} " + " ".join(f"{k[:8]:>8s}" for k in kinds)
    print(head)
    results = []
    for f in files:
        r = lint_file(f)
        results.append(r)
        flags = " MONOTONE" if r["monotone"] else ""
        print(f"{f.name:22s} {r['score']:5.0f} {r['words']:5d} {r['sentences']:4d} {r['mean_len']:5.1f} {r['stdev_len']:5.1f} {r['we_density']:6.2f} "
              + " ".join(f"{r['counts'].get(k, 0):8d}" for k in kinds) + flags)
    print()
    for r in results:
        if not r["hits"]:
            continue
        rel = r["path"].relative_to(ROOT) if r["path"].is_absolute() and ROOT in r["path"].resolve().parents else r["path"]
        print(f"-- {rel}  (score {r['score']:.0f}; penalty {r['penalty']:.1f} from patterns, {r['we_pen']:.1f} from 'we'"
              + (f", {MONOTONE_PENALTY} monotone" if r["monotone"] else "") + ")")
        if r["bold_per_section"]:
            print("   bold lead-ins per section: " + "; ".join(f"{s}: {n}" for s, n in r["bold_per_section"].items()))
        for kind, line, snippet in sorted(r["hits"], key=lambda h: (h[1], h[0])):
            print(f"   {kind:9s} L{line:<4d} {snippet[:150]}")
        print()
    # worst offending passages verbatim, across files, heaviest patterns first (a source line here is a
    # whole paragraph, so the offending sentence is printed with the paragraph's line number)
    ranked = sorted((-WEIGHTS[kind], r["path"].name, line, kind, snippet) for r in results for kind, line, snippet in r["hits"])
    print(f"== worst {min(a.top, len(ranked))} offending passages (verbatim, with the source line of their paragraph) ==")
    for w, name, line, kind, text in ranked[:a.top]:
        print(f"[{kind} w={-w}] {name}:{line}\n    {text[:400]}")
    low = [r for r in results if r["score"] < a.min]
    print(f"\n== {len(results)} files; {sum(len(r['hits']) for r in results)} findings; {len(low)} file(s) below {a.min:.0f} ==")
    sys.exit(1 if a.strict and low else 0)


if __name__ == "__main__":
    main()
