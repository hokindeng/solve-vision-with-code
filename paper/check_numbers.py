#!/usr/bin/env python3
"""Every number in the paper must trace to the released result files.

    python3 paper/check_numbers.py          # run from the repo root; exit 1 on any failure
    python3 paper/check_numbers.py --pdf    # also inspect paper/main.pdf (needs pdftotext)
    python3 paper/check_numbers.py --quiet  # only failures, warnings and the summary

Sources of truth (read only):
    bench/paper/table1_leaderboard.csv        37 systems: overall / in-domain / out-of-domain
    bench/paper/table2_categories.csv         coding agents x five categories
    bench/paper/table3_efficiency.csv         run statistics per lane (from bench/stats.json)
    bench/paper/table4_per_task_closed.csv    per-task means of the three closed-model agents
    bench/stats.json                          run statistics per lane, incl. score over produced videos
    bench/results/<lane>/all_models_summary.json and <lane>_vbvr_results.json (per-instance scores)

Checks
    (a) every three-decimal number in paper/sec/*.tex and paper/tables/*.tex resolves to a source value,
        a difference of two source values, or an explicitly whitelisted external number; two-decimal
        numbers are resolved the same way (rounded to two decimals) and fail when unresolved
    (b) the headline numbers (top-3 agents, best video model, best open-weight model) are in the abstract
    (c) counts in prose ("24 open-weight models", "100 tasks", ...) are listed for a human; the
        open-weight count is compared with table1 and the roster (report only)
    (d) the generated tables tab_main, tab_categories, tab_efficiency, tab_hardest are recomputed from
        the CSVs and compared cell by cell
    --pdf: table captions 2..5 precede "6. Conclusion", no draft markers, running header on page 2,
        page count

Standard library only.
"""
import argparse
import csv
import itertools
import json
import re
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
CSV_DIR = ROOT / "bench" / "paper"
RESULTS = ROOT / "bench" / "results"

CATS = ["Abstraction", "Perception", "Spatiality", "Transformation", "Knowledge"]
# lane directory -> CSV model name, for the three closed-model agents (per-instance results are read for these)
CLOSED = {
    "codex-gpt-6-astra": "Codex/gpt-6-astra",
    "claude-fable-5.1-bedrock": "Claude Code/Fable 5.1",
    "gemini-gemini-3.1-pro-preview": "Gemini CLI/3.1 Pro",
}
# CSV model name -> display name used in the .tex tables (same map as paper/figures/make_figures.py)
DISPLAY = {
    "Codex/gpt-6-astra": "Codex (gpt-6-astra)",
    "Claude Code/Fable 5.1": "Claude Code (Fable 5.1)",
    "Gemini CLI/3.1 Pro": "Gemini CLI (Gemini 3.1 Pro)",
    "VBVR-Pro-Wan2.2-I2V-A14B (RL-trained video model)": "VBVR-Pro-Wan2.2-I2V-A14B (RL)",
}

# External numbers that have no source in this repository. Each entry: value -> provenance.
# VBVR-Pro paper, Table 8: in-domain category means of the released baseline VBVR-Pro-Wan2.2-TI2V-5B.
# verified against arXiv 2608.26105 Table 8 on 2026-09-17
TABLE8_VIDEO_BASELINE = {
    "Abstraction": 0.578,
    "Perception": 0.476,
    "Spatiality": 0.480,
    "Transformation": 0.724,
    "Knowledge": 0.511,
    # its in-domain overall (0.641) is also the ID column of table1_leaderboard.csv and is not whitelisted here
}
# Two-decimal constants of the protocol and of the source tasks (appendix), not results. Kept explicit so a
# new unexplained two-decimal number still fails.
WHITELIST_2DP = {
    "0.95": "reward-check threshold: ground truth >= 0.95 (appendix, tab_reward_check)",
    "0.05": "reward shaping: program crashes (appendix, training setup)",
    "0.10": "reward shaping: runs without writing a video (appendix, training setup)",
    "0.15": "reward shaping: 0.15 + 0.85 x evaluator score (appendix, training setup)",
    "0.85": "reward shaping: 0.15 + 0.85 x evaluator score (appendix, training setup)",
    "0.86": "source-task defect O-9: 200% enlargement renders as a 0.86 shrink (appendix, limitations)",
    "0.80": "source-task defect O-62: prompt elasticity 0.80 vs simulation 0.7 (appendix, limitations)",
}

NUM_RE = re.compile(r"(?<![\d.])(?:0\.\d{3}|1\.000|0\.\d{2})(?![\d.]|\\)")
COUNT_RE = re.compile(r"\b(\d{1,3}) (tasks|families|instances|models|lanes)\b")


# ---------------------------------------------------------------- helpers
def r3(v):
    return f"{float(v):.3f}"


def r2(v):
    return f"{float(v):.2f}"


def strip_comment(line):
    """Drop a LaTeX comment (a % not preceded by a backslash) and length arguments such as 0.62\\columnwidth."""
    line = re.sub(r"(?<!\\)%.*", "", line)
    line = re.sub(r"\\includegraphics\[[^\]]*\]", r"\\includegraphics", line)
    line = re.sub(r"\d*\.?\d+\\(?:textwidth|columnwidth|linewidth|textheight)", "", line)
    return line


def read_csv(name):
    with open(CSV_DIR / name, newline="") as f:
        return list(csv.DictReader(f))


def display_name(csv_name):
    if csv_name in DISPLAY:
        return DISPLAY[csv_name]
    if csv_name.startswith("OpenCode × "):
        return csv_name.replace("OpenCode × ", "").split(" (")[0]
    return csv_name


def per_instance(lane):
    p = RESULTS / lane / f"{lane}_vbvr_results.json"
    return json.load(open(p))["samples"]


def tex_cells(line):
    """Split a tabular data row into plain cells: drop \\rowcolor, \\makecell, \\textbf, \\underline, \\emph, $^\\dagger$, trailing \\\\."""
    line = re.sub(r"\\rowcolor\{[^}]*\}\s*", "", line)
    for _ in range(3):  # formatting commands, innermost first
        line = re.sub(r"\\(?:textbf|underline|emph|textit|texttt)\{([^{}]*)\}", r"\1", line)
    line = re.sub(r"\\makecell(?:\[[^\]]*\])?\{([^{}]*)\}", lambda m: re.sub(r"\\\\\s*", " ", m.group(1)), line)  # line-broken cell -> one line
    line = re.sub(r"\\\\(\[[^\]]*\])?\s*$", "", line.strip())
    line = line.replace("$^\\dagger$", "").replace("\\_", "_").replace("$\\times$", "×")
    return [c.strip() for c in line.split("&")]


class Report:
    def __init__(self, quiet=False):
        self.quiet = quiet
        self.fail = 0
        self.warn = 0
        self.lines = []

    def out(self, s="", always=False):
        if always or not self.quiet:
            print(s)

    def failure(self, s):
        self.fail += 1
        print("FAIL  " + s)

    def warning(self, s):
        self.warn += 1
        print("WARN  " + s)


# ---------------------------------------------------------------- sources
class Entry:
    """One source value with its provenance tags."""
    __slots__ = ("pool", "value", "system", "column", "task", "prov")

    def __init__(self, pool, value, prov, system=None, column=None, task=None):
        self.pool, self.value, self.prov, self.system, self.column, self.task = pool, float(value), prov, system, column, task


class Sources:
    """Every source value as an Entry, indexed by pool and by rounded value; differences of pairs; aliases."""

    def __init__(self):
        self.entries = []
        self.t1 = read_csv("table1_leaderboard.csv")
        self.t2 = read_csv("table2_categories.csv")
        self.t3 = read_csv("table3_efficiency.csv")
        self.t4 = read_csv("table4_per_task_closed.csv")
        self.stats = json.load(open(ROOT / "bench" / "stats.json"))
        self.notes = []
        self._build()
        self._index()
        self._aliases()

    def add(self, pool, value, prov, system=None, column=None, task=None):
        self.entries.append(Entry(pool, value, prov, system, column, task))

    def _build(self):
        lane_of_name = {n: l for l, n in CLOSED.items()}
        for r in self.t1:
            for col in ("overall", "in_domain", "out_of_domain"):
                self.add("table1", r[col], f"table1 {r['model']} {col}={r[col][:7]}", r["model"], col)
        for r in self.t2:
            for col in CATS + ["overall"]:
                self.add("table2", r[col], f"table2 {r['model']} {col}={r[col][:7]}", r["model"], col)
        for r in self.t3:
            for col in ("score", "produced_rate", "tool_use_rate"):
                self.add("stats", r[col], f"table3 {r['model']} {col}={r[col][:7]}", r["model"], col)
        # stats.json lanes are keyed by directory; map them to the CSV model names through table3's order-free match on score
        by_score = {round(float(r["score"]), 12): r["model"] for r in self.t3 if float(r["score"]) > 0}
        for lane, st in self.stats.items():
            system = CLOSED.get(lane) or by_score.get(round(st["score_over_500"], 12), lane)
            for col in ("score_over_500", "score_over_produced", "produced_rate", "tool_use_rate"):
                self.add("stats", st[col], f"stats.json {lane} {col}={str(st[col])[:7]}", system, col)
        for r in self.t4:
            lanes = [c for c in r if c not in ("task", "split")]
            for col in lanes:
                self.add("table4", r[col], f"table4 {r['task']} {col}={r[col][:7]}", col, "per-task", r["task"].split("_")[0])
            m = statistics.mean(float(r[c]) for c in lanes)
            self.add("table4", m, f"table4 {r['task']} mean of the three agents={m:.4f}", "closed agents", "per-task mean", r["task"].split("_")[0])
        # per-instance results of the closed three: split-restricted category means and the instance scores
        self.id_cat, self.ood_cat = {}, {}
        self.n_tasks, self.n_tasks_id = defaultdict(set), defaultdict(set)
        self.category_of = {}
        for lane, name in CLOSED.items():
            s = per_instance(lane)
            for x in s:
                t = x["task_name"].replace("_data-generator", "")
                self.category_of[t] = x["category"]
                self.n_tasks[x["category"]].add(t)
                if x["split"] == "In_Domain":
                    self.n_tasks_id[x["category"]].add(t)
                self.add("per-instance", x["score"], f"{lane} {t} {x['video_file']} score={x['score']:.4f}", name, "instance", t.split("_")[0])
            self.id_cat[name], self.ood_cat[name] = {}, {}
            for c in CATS:
                for split, store in (("In_Domain", self.id_cat), ("Out_of_Domain", self.ood_cat)):
                    xs = [x["score"] for x in s if x["category"] == c and x["split"] == split]
                    m = sum(xs) / len(xs)
                    store[name][c] = m
                    self.add("category", m, f"{name} {split} {c} mean over {len(xs)} instances={m:.4f}", name, c, split)
            summ = json.load(open(RESULTS / lane / "all_models_summary.json"))[lane]
            t1 = next(r for r in self.t1 if r["model"] == name)
            t2 = next(r for r in self.t2 if r["model"] == name)
            ok = (abs(summ["overall"] - float(t1["overall"])) < 1e-9 and abs(summ["In_Domain"] - float(t1["in_domain"])) < 1e-9
                  and abs(summ["Out_of_Domain"] - float(t1["out_of_domain"])) < 1e-9
                  and all(abs(summ["by_category"][c] - float(t2[c])) < 1e-9 for c in CATS))
            self.notes.append(f"{'ok  ' if ok else 'MISMATCH'} all_models_summary.json({lane}) == table1/table2" + ("" if ok else "  <-- FAIL"))
        for c in CATS:
            for r in self.t2:
                self.add("category", r[c], f"table2 {r['model']} {c}={r[c][:7]}", r["model"], c, "both splits")
        for c, v in TABLE8_VIDEO_BASELINE.items():
            self.add("table8", v, f"VBVR-Pro Table 8 {c}={v} (external, whitelisted)", "VBVR-Pro-Wan2.2-TI2V-5B", c)
            self.add("category", v, f"VBVR-Pro Table 8 {c}={v} (external)", "VBVR-Pro-Wan2.2-TI2V-5B", c, "In_Domain")
        # reward-check table: external numbers (fresh-seed reward validation) whose only in-repo record is the table
        rc = PAPER / "tables" / "tab_reward_check.tex"
        if rc.exists():
            for i, line in enumerate(rc.read_text().splitlines(), 1):
                for m in NUM_RE.finditer(strip_comment(line)):
                    self.add("reward-check", m.group(0), f"tab_reward_check.tex:{i} (external table, no CSV source)")

    def _index(self):
        self.by3 = defaultdict(list)
        for e in self.entries:
            self.by3[r3(e.value)].append(e)
        # differences: table1 and stats, any two values of the pool; category pool, one category and one split (or one system).
        # Rank 0: one system, its natural column pair (ID/OOD; score over 500 / over produced). Rank 1: one
        # column, two systems. Rank 2: one system, other columns. Rank 3: unrelated.
        natural = {frozenset(("in_domain", "out_of_domain")), frozenset(("score_over_500", "score_over_produced"))}
        self.diffs = defaultdict(list)  # rounded3 -> [(rank, a, b, d, rounded_only)]
        for prio, pool in enumerate(("table1", "stats", "category")):
            es = [e for e in self.entries if e.pool == pool]
            for a, b in itertools.combinations(es, 2):
                d = abs(a.value - b.value)
                if d < 5e-4 or (pool == "category" and (a.column != b.column or (a.task != b.task and a.system != b.system))):
                    continue  # category differences: one category, and either one split or one system
                rank = 0 if a.system == b.system and frozenset((a.column, b.column)) in natural else 1 if a.column == b.column else 2 if a.system == b.system else 3
                self.diffs[r3(d)].append((rank, prio, a, b, d, False))
                dr = abs(round(a.value, 3) - round(b.value, 3))
                if r3(dr) != r3(d):
                    self.diffs[r3(dr)].append((rank + 4, prio, a, b, d, True))
        for k in self.diffs:
            self.diffs[k].sort(key=lambda t: (t[0], t[1]))
        self.by2 = defaultdict(list)
        for e in self.entries:
            self.by2[r2(e.value)].append(e)
        self.diffs2 = defaultdict(list)
        for k, lst in self.diffs.items():
            self.diffs2[r2(k)].extend(lst)

    def _aliases(self):
        """alias text -> set of CSV model names it may refer to (longest alias wins at a position)."""
        al = defaultdict(set)
        for r in self.t1:
            n = r["model"]
            al[n].add(n)
            al[display_name(n)].add(n)
            for part in re.split(r"/| \(", display_name(n).replace(")", "")):
                part = part.strip()
                if len(part) >= 4 and not re.fullmatch(r"(?:RL|video model|coding agent.*)", part):
                    al[part].add(n)
        hand = {
            "Codex": "Codex/gpt-6-astra", "Claude Code": "Claude Code/Fable 5.1", "Claude": "Claude Code/Fable 5.1", "Fable": "Claude Code/Fable 5.1",
            "Gemini CLI": "Gemini CLI/3.1 Pro", "Gemini": "Gemini CLI/3.1 Pro",
            "A14B": "VBVR-Pro-Wan2.2-I2V-A14B (RL-trained video model)", "RL-trained": "VBVR-Pro-Wan2.2-I2V-A14B (RL-trained video model)",
            "best video model": "VBVR-Pro-Wan2.2-I2V-A14B (RL-trained video model)", "TI2V-5B": "VBVR-Pro-Wan2.2-TI2V-5B", "baseline": "VBVR-Pro-Wan2.2-TI2V-5B",
            "LTX2.3": "VBVR-Pro-LTX2.3", "Seedance": "Seedance 2.0", "Kling": "Kling VIDEO 3.0", "Veo": "Veo 3.1",
            "MiniMax": "MiniMax M2.5", "DeepSeek": "DeepSeek V3.2", "Devstral": "Devstral 2 123B", "Nemotron Super": "Nemotron Super 3 120B",
            "Nemotron Nano": "Nemotron Nano 3 30B", "Mistral Large": "Mistral Large 3 675B", "Qwen3-VL": "Qwen3-VL-235B-A22B",
            "Qwen3-Next": "Qwen3-Next-80B-A3B", "Qwen3-Coder-30B": "Qwen3-Coder-30B-A3B", "Llama 3.3": "OpenCode × Llama 3.3 70B (open weights, Bedrock via converse proxy)",
            "Magistral": "OpenCode × Magistral Small (open weights, Bedrock)", "Gemma 3 27B": "OpenCode × Gemma 3 27B (open weights, Bedrock)",
            "Gemma 3 12B": "OpenCode × Gemma 3 12B (open weights, Bedrock)",
        }
        for a, n in hand.items():
            al[a].add(n)
        closed = set(CLOSED.values())
        for a in ("three agents", "three closed", "closed agents", "closed-model agents", "closed-model coding agents", "the agents", "all three",
                  "best agent", "best coding agent", "strongest agent", "top three", "top-3", "best score", "agents"):
            al[a] |= closed
        rows = sorted(self.t1, key=lambda r: -float(r["overall"]))
        best_open = next(r["model"] for r in rows if "open" in r["kind"])
        for a in ("best open", "best open-weight", "best of them", "the best"):
            al[a].add(best_open)
        al["the best"] |= {rows[0]["model"], next(r["model"] for r in rows if r["kind"] == "video model")}
        for tag in ("Wan2.2", "Wan2.1", "LTX", "Kimi", "gpt-oss", "Qwen3", "Llama 4", "Gemma"):  # ambiguous family names
            for r in self.t1:
                if tag.lower() in r["model"].lower():
                    al[tag].add(r["model"])
        # video-model group words: any video model
        for word in ("video model", "video models", "trained video"):
            al[word] |= {r["model"] for r in self.t1 if r["kind"] == "video model"}
        self.aliases = sorted(al.items(), key=lambda kv: -len(kv[0]))
        self.lane_alias = {"codex": "Codex/gpt-6-astra", "claude": "Claude Code/Fable 5.1", "gemini": "Gemini CLI/3.1 Pro"}

    def named(self, text):
        """Systems, tasks and categories named in a piece of text."""
        systems, tasks, cats = set(), set(), set()
        scan = text
        for alias, names in self.aliases:
            if alias in scan:
                systems |= names
                scan = scan.replace(alias, " ")
        for m in re.finditer(r"\b([GO]-\d+)\b", text):
            tasks.add(m.group(1))
        for c in CATS:
            if c.lower() in text.lower():
                cats.add(c)
        for lane_word, name in self.lane_alias.items():  # table columns named Codex / Claude / Gemini
            if re.search(rf"\b{lane_word}\b", text, re.I):
                systems.add(name)
        return systems, tasks, cats

    def context(self, text, pos):
        """The sentence around position pos of text (a .tex line is a paragraph)."""
        left = text[:pos]
        cut = max([m.end() for m in re.finditer(r"(?<=[.;!?])\s+(?=[A-Z\\(])", left)] + [0])
        right = text[pos:]
        m = re.search(r"[.;!?](?=\s+[A-Z\\(]|\s*$)", right)
        return left[cut:] + (right[:m.end()] if m else right)

    def _related(self, e, systems, tasks, cats):
        if e.pool in ("table8", "reward-check"):
            return True
        if e.pool in ("table4", "per-instance"):
            return e.task in tasks
        if e.pool == "category":
            return e.system in systems and (e.column in cats or not cats)
        return e.system in systems

    def resolve(self, token, text, pos, extra=()):
        """-> (status, provenance).
        Strong: OK / DIFF / DIFFr / EXT / INST / OK2 / DIFF2 / WL2, resolved against a system, task or category named in the
        same sentence; a trailing '~' on OK/OK2/INST means it was named only elsewhere in the paragraph or in the table header.
        Weak (WARN; FAIL with --strict): OK? / DIFF? / INST? / OK2? / DIFF2?. FAIL: nothing resolves."""
        two = len(token) == 4
        if token in ("1.000", "0.000", "1.00", "0.00"):
            return ("OK2" if two else "OK"), "degenerate value (0 or 1)"
        if two and token in WHITELIST_2DP:
            return "WL2", WHITELIST_2DP[token]
        cands = self.by2[token] if two else self.by3[token]
        dcands = self.diffs2[token] if two else self.diffs[token]

        def desc(d):
            rank, _, a, b, dv, rounded = d
            tag = {0: "", 1: "[same column] ", 2: "[same system, other columns] ", 3: "[unrelated pair] "}[rank % 4]
            la, lb = a.prov.split("=")[0].strip(), b.prov.split("=")[0].strip()
            return tag + (f"|{r3(a.value)} - {r3(b.value)}| ({la} - {lb}; raw difference {dv:.4f} rounds elsewhere)" if rounded else f"|{la} - {lb}| = {dv:.4f}")

        def short(names):
            names = sorted(display_name(n) for n in names)
            return names if len(names) <= 4 else names[:4] + [f"+{len(names) - 4} more"]

        tiers = [("", self.named(self.context(text, pos)))]
        para = self.named(text)
        tiers.append(("~", tuple(a | b for a, b in zip(para, extra)) if extra else para))
        for mark, (systems, tasks, cats) in tiers:
            strong = [e for e in cands if e.pool != "per-instance" and self._related(e, systems, tasks, cats)]
            if strong:
                e = strong[0]
                st = "EXT" if e.pool in ("table8", "reward-check") else "OK2" if two else "OK"
                return st + (mark if st != "EXT" else ""), e.prov + (f" (+{len(strong) - 1} more)" if len(strong) > 1 else "")
            strong_d = [d for d in dcands if d[2].system in systems and d[3].system in systems]
            if strong_d:
                d = strong_d[0]
                return ("DIFF2" if two else "DIFFr" if d[5] else "DIFF") + mark, desc(d)
            inst = [e for e in cands if e.pool == "per-instance" and e.task in tasks]
            if inst:
                return "INST" + mark, inst[0].prov + f" (per-instance score; {len(inst)} instance(s) of the named task match)"
        systems = tiers[0][1][0]
        weak = [e for e in cands if e.pool != "per-instance"]
        if weak:
            e = weak[0]
            return ("OK2?" if two else "OK?"), f"{e.prov}  <- nothing in the sentence names its system/task (sentence names {short(systems) or 'no system'}); {len(weak)} candidate(s)"
        if dcands:
            return ("DIFF2?" if two else "DIFF?"), desc(dcands[0]) + f"  <- the systems are not named in the sentence; {len(dcands)} candidate pair(s)"
        inst = [e for e in cands if e.pool == "per-instance"]
        if inst:
            return "INST?", inst[0].prov + f"  <- task not named; {len(inst)} instance(s) match"
        return "FAIL", "no source value, difference or whitelist entry"


# ---------------------------------------------------------------- (a) numbers in the prose and tables
WEAK = {"OK?", "DIFF?", "INST?", "OK2?", "DIFF2?"}


def check_numbers(src, rep, strict=False):
    files = sorted((PAPER / "sec").glob("*.tex")) + sorted((PAPER / "tables").glob("*.tex"))
    rep.out("== (a) numbers in paper/sec/*.tex and paper/tables/*.tex ==", always=True)
    rep.out(f"{'file:line':34s} {'number':7s} {'status':6s} resolved to")
    counts = defaultdict(int)
    for f in files:
        rel = f.relative_to(ROOT)
        lines = f.read_text().splitlines()
        extra = ()
        if f.parent.name == "tables":  # the header rows name the column lanes; they count for every row of the table
            head = "\n".join(strip_comment(l) for l in lines[:next((k for k, l in enumerate(lines) if "\\midrule" in l), len(lines))])
            extra = src.named(head)
        for i, line in enumerate(lines, 1):
            clean = strip_comment(line)
            for m in NUM_RE.finditer(clean):
                tok = m.group(0)
                status, prov = src.resolve(tok, clean, m.start(), extra)
                counts[status] += 1
                row = f"{str(rel) + ':' + str(i):34s} {tok:7s} {status:6s} {prov}"
                if status == "FAIL" or (strict and status in WEAK):
                    rep.failure(row)
                elif status in WEAK:
                    rep.warning(row)
                else:
                    rep.out(row)
    rep.out("status counts: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())), always=True)
    rep.out("  strong: OK source value of a system/task named in the sentence | DIFF difference of two named systems' values (DIFFr via rounded values)\n"
            "          EXT external (Table 8 whitelist / reward-check table) | INST per-instance score of the named task\n"
            "          OK2 two-decimal rounding of a named system's value | DIFF2 two-decimal rounding of such a difference | WL2 whitelisted constant\n"
            "          a trailing ~ (OK~, DIFF~, INST~): named only elsewhere in the paragraph or in the table header\n"
            "  weak (WARN; FAIL with --strict): OK? / DIFF? / INST? / OK2? / DIFF2? — a source value exists but nothing nearby ties the number to it", always=True)


# ---------------------------------------------------------------- (b) headline numbers in the abstract
def check_abstract(src, rep):
    rep.out("\n== (b) headline numbers in the abstract ==", always=True)
    text = "\n".join(strip_comment(l) for l in (PAPER / "sec" / "0_abstract.tex").read_text().splitlines())
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", text, re.S)
    if not m:
        rep.failure("no \\begin{abstract} ... \\end{abstract} in paper/sec/0_abstract.tex")
        return
    abstract = m.group(1)
    rows = sorted(src.t1, key=lambda r: -float(r["overall"]))
    agents = [r for r in rows if r["kind"] != "video model"]
    want = [(f"agent #{k + 1} {r['model']}", r3(r["overall"])) for k, r in enumerate(agents[:3])]
    best_video = max((r for r in rows if r["kind"] == "video model"), key=lambda r: float(r["overall"]))
    best_open = max((r for r in rows if "open" in r["kind"]), key=lambda r: float(r["overall"]))
    want += [(f"best video model {best_video['model']}", r3(best_video["overall"])), (f"best open-weight {best_open['model']}", r3(best_open["overall"]))]
    for label, v in want:
        if v in abstract:
            rep.out(f"ok    {v}  {label}")
        else:
            rep.failure(f"abstract lacks {v} ({label})")


# ---------------------------------------------------------------- (c) counts in prose
def check_counts(src, rep):
    rep.out("\n== (c) counts in prose (for a human) ==", always=True)
    open_rows = [r for r in src.t1 if "open" in r["kind"]]
    open_scored = [r for r in open_rows if float(r["overall"]) > 0]
    roster_n = None
    roster = ROOT / "bench" / "roster_bedrock_open.json"
    if roster.exists():
        cells = json.load(open(roster)).get("cells", {})
        roster_n = len(cells)
        unsupported = [k for k, v in cells.items() if isinstance(v, dict) and v.get("route") == "unsupported"]
    rep.out(f"table1: {len(src.t1)} systems, {len(open_rows)} open-weight rows ({len(open_scored)} with a video, {len(open_rows) - len(open_scored)} zero lanes), "
            f"{sum(1 for r in src.t1 if r['kind'] == 'video model')} video models; roster: {roster_n} models"
            + (f" ({len(unsupported)} unsupported: {', '.join(unsupported)})" if roster_n else ""), always=True)
    seen = defaultdict(list)
    for f in sorted((PAPER / "sec").glob("*.tex")):
        rel = f.relative_to(ROOT)
        for i, line in enumerate(f.read_text().splitlines(), 1):
            clean = strip_comment(line)
            for m in COUNT_RE.finditer(clean):
                ctx = clean[max(0, m.start() - 30):m.end() + 20].replace("\n", " ")
                seen[m.group(0)].append(f"{rel}:{i}  …{ctx}…")
    for phrase, where in sorted(seen.items(), key=lambda kv: (kv[0].split()[1], -len(kv[1]))):
        rep.out(f"{phrase:22s} x{len(where):<3d} {where[0]}")
        for w in where[1:]:
            rep.out(f"{'':26s} {w}")
    # open-weight count consistency
    ow = defaultdict(list)
    for f in sorted((PAPER / "sec").glob("*.tex")):
        for i, line in enumerate(f.read_text().splitlines(), 1):
            for m in re.finditer(r"\b(\d{1,3}) open-weight\b", strip_comment(line)):
                ow[int(m.group(1))].append(f"{f.relative_to(ROOT)}:{i}")
    if ow:
        summary = "; ".join(f"'{n} open-weight' x{len(w)} ({', '.join(w)})" for n, w in sorted(ow.items()))
        if len(ow) > 1:
            rep.warning(f"open-weight count is inconsistent: {summary}. table1 has {len(open_rows)} open-weight rows; the roster has {roster_n}.")
        else:
            rep.out(f"open-weight count consistent: {summary}; table1 open rows={len(open_rows)}, roster={roster_n}", always=True)


# ---------------------------------------------------------------- (d) generated tables, cell by cell
def data_rows(path):
    """(lineno, cells) for every tabular data row of a table .tex file."""
    out = []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        s = strip_comment(line).strip()
        if "&" not in s or "multicolumn" in s:
            continue
        out.append((i, tex_cells(s)))
    return out


def compare(rep, name, lineno, expected, actual):
    if expected == actual:
        return True
    rep.failure(f"{name}:{lineno}\n        expected: {' & '.join(expected)}\n        in file : {' & '.join(actual)}")
    return False


def check_tab_main(src, rep):
    name = "paper/tables/tab_main.tex"
    rows = sorted(src.t1, key=lambda r: -float(r["overall"]))
    rank = {r["model"]: i for i, r in enumerate(rows, 1)}
    produced = {r["model"]: int(r["produced"]) for r in src.t3}
    kind = lambda r: "video" if r["kind"] == "video model" else "open" if "open" in r["kind"] else "closed"
    expected = []
    for k in ("closed", "open", "video"):
        for r in [r for r in rows if kind(r) == k]:
            if k == "video":
                vids = "---"
            else:
                p = produced.get(r["model"], produced.get(display_name(r["model"])))
                vids = "?" if p is None else str(p)
            expected.append([str(rank[r["model"]]), display_name(r["model"]), r3(r["overall"]), r3(r["in_domain"]), r3(r["out_of_domain"]), vids])
    actual = data_rows(PAPER / "tables" / "tab_main.tex")
    actual = [(i, c) for i, c in actual if len(c) == 6 and c[0].isdigit()]
    ok = len(actual) == len(expected)
    if not ok:
        rep.failure(f"{name}: {len(actual)} data rows, expected {len(expected)}")
    for (i, cells), exp in zip(actual, expected):
        ok &= compare(rep, name, i, exp, cells)
    rep.out(f"{'ok  ' if ok else 'FAIL'} {name}: {len(actual)} rows x 6 cells recomputed from table1 + table3 (videos)", always=True)


def check_tab_categories(src, rep):
    name = "paper/tables/tab_categories.tex"
    cats = sorted(src.t2, key=lambda r: -float(r["overall"]))
    closed = [r for r in cats if r["model"] in CLOSED.values()]
    opens = [r for r in cats if r["model"] not in CLOSED.values()][:8]
    expected = [[display_name(r["model"])] + [r3(r[c]) for c in CATS] + [r3(r["overall"])] for r in closed + opens]
    id_of = {r["model"]: r["in_domain"] for r in src.t1}
    for r in closed:
        expected.append([display_name(r["model"])] + [r3(src.id_cat[r["model"]][c]) for c in CATS] + [r3(id_of[r["model"]])])
    v = TABLE8_VIDEO_BASELINE
    expected.append(["VBVR-Pro-Wan2.2-TI2V-5B (video model)"] + [r3(v[c]) for c in CATS] + [r3(id_of["VBVR-Pro-Wan2.2-TI2V-5B"])])
    header_all = ["Tasks per category"] + [f"({len(src.n_tasks[c])})" for c in CATS] + ["(100)"]
    header_id = ["Tasks per category"] + [f"({len(src.n_tasks_id[c])})" for c in CATS] + ["(50)"]
    actual = data_rows(PAPER / "tables" / "tab_categories.tex")
    heads = [(i, c) for i, c in actual if c and c[0] == "Tasks per category"]
    body = [(i, c) for i, c in actual if len(c) == 7 and c[0] not in ("System", "Tasks per category")]
    ok = True
    if len(heads) != 2:
        rep.failure(f"{name}: expected two 'Tasks per category' header rows, found {len(heads)}")
        ok = False
    else:
        ok &= compare(rep, name, heads[0][0], header_all, heads[0][1])
        ok &= compare(rep, name, heads[1][0], header_id, heads[1][1])
    if len(body) != len(expected):
        rep.failure(f"{name}: {len(body)} data rows, expected {len(expected)}")
        ok = False
    for (i, cells), exp in zip(body, expected):
        ok &= compare(rep, name, i, exp, cells)
    rep.out(f"{'ok  ' if ok else 'FAIL'} {name}: {len(body)} rows x 7 cells recomputed from table2, per-instance results (ID block), table1 (ID overall), Table 8", always=True)


def check_tab_efficiency(src, rep):
    name = "paper/tables/tab_efficiency.tex"
    rows = sorted(src.t3, key=lambda r: -float(r["score"]))
    closed = [r for r in rows if r["model"] in CLOSED.values()]
    opens = [r for r in rows if r["model"] not in CLOSED.values() and float(r["score"]) > 0][:6]
    expected = []
    for r in closed + opens:
        expected.append([display_name(r["model"]), r3(r["score"]), r["produced"], r2(r["tool_use_rate"]), f"{float(r['tool_calls_per_attempt']):.1f}",
                         f"{float(r['median_seconds']):.0f}", f"{float(r['mean_input_tokens']) / 1000:.0f}", f"{float(r['mean_output_tokens']) / 1000:.1f}", r["timeouts"]])
    actual = [(i, c) for i, c in data_rows(PAPER / "tables" / "tab_efficiency.tex") if len(c) == 9 and c[0] != "Agent"]
    ok = len(actual) == len(expected)
    if not ok:
        rep.failure(f"{name}: {len(actual)} data rows, expected {len(expected)}")
    for (i, cells), exp in zip(actual, expected):
        ok &= compare(rep, name, i, exp, cells)
    rep.out(f"{'ok  ' if ok else 'FAIL'} {name}: {len(actual)} rows x 9 cells recomputed from table3 (score .3f, tool use .2f, calls .1f, median s .0f, tokens/1000 .0f/.1f)", always=True)


def check_tab_hardest(src, rep):
    name = "paper/tables/tab_hardest.tex"
    lanes = [c for c in src.t4[0] if c not in ("task", "split")]
    pt = sorted(src.t4, key=lambda r: statistics.mean(float(r[l]) for l in lanes))
    expected = []
    for r in pt[:12]:
        tid, _, rest = r["task"].partition("_")
        split = {"In_Domain": "ID", "Out_of_Domain": "OOD"}.get(r["split"], r["split"])
        vals = [float(r[l]) for l in lanes]
        expected.append([tid, rest.replace("_", " "), split, src.category_of.get(r["task"], "?")] + [r2(v) for v in vals] + [r2(statistics.mean(vals))])
    actual = [(i, c) for i, c in data_rows(PAPER / "tables" / "tab_hardest.tex") if len(c) == 8 and c[0] != "Task"]
    ok = len(actual) == len(expected)
    if not ok:
        rep.failure(f"{name}: {len(actual)} data rows, expected {len(expected)}")
    for (i, cells), exp in zip(actual, expected):
        ok &= compare(rep, name, i, exp, cells)
    rep.out(f"{'ok  ' if ok else 'FAIL'} {name}: {len(actual)} rows x 8 cells recomputed from table4 (12 lowest three-agent means; category from the evaluator results)", always=True)


# ---------------------------------------------------------------- --pdf
def check_pdf(rep):
    rep.out("\n== --pdf: paper/main.pdf ==", always=True)
    pdf = PAPER / "main.pdf"
    if not pdf.exists():
        rep.failure("paper/main.pdf missing")
        return
    if not shutil.which("pdftotext"):
        rep.failure("pdftotext not on PATH (poppler)")
        return
    run = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True)
    if run.returncode or not run.stdout.strip():
        rep.failure(f"pdftotext could not read paper/main.pdf (exit {run.returncode}; file being rebuilt?): {run.stderr.strip()[:200]}")
        return
    text = run.stdout
    pages = text.split("\f")
    if pages and not pages[-1].strip():
        pages = pages[:-1]
    lines = text.splitlines()
    rep.out(f"pages: {len(pages)}", always=True)

    def norm(s):
        s = re.sub(r"\\(?:emph|textbf|texttt|cref|Cref|ref|citep|citet|cite)(\[[^\]]*\])?\{([^{}]*)\}", r"\2", s)
        s = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^{}]*\})?", " ", s)
        return re.sub(r"[^a-z0-9]", "", s.lower())

    # captions from the .tex sources: their first 40 normalised characters identify the caption line in the PDF text
    captions = []
    for f in list((PAPER / "tables").glob("*.tex")) + list((PAPER / "sec").glob("*.tex")):
        src = "\n".join(strip_comment(l) for l in f.read_text().splitlines())
        for m in re.finditer(r"\\caption\{", src):
            depth, j = 1, m.end()
            while j < len(src) and depth:
                depth += {"{": 1, "}": -1}.get(src[j], 0)
                j += 1
            captions.append(norm(src[m.end():j - 1])[:40])
    caption_line = {}
    for i, l in enumerate(lines):
        m = re.match(r"^\s*(Table|Figure)\s+(\d+)[.:]?\s+(.*)", l)
        if m and any(norm(m.group(3)).startswith(c[:25]) for c in captions if len(c) >= 25):
            caption_line.setdefault(f"{m.group(1)} {m.group(2)}", i)
    concl = next((i for i, l in enumerate(lines) if re.match(r"^\s*6\.\s+Conclusion\b", l)), None)
    if concl is None:
        rep.failure("'6. Conclusion' heading not found in the PDF text")
    else:
        rep.out(f"'6. Conclusion' at page {text[:sum(len(x) + 1 for x in lines[:concl])].count(chr(12)) + 1}", always=True)
        for k in range(2, 6):
            key = f"Table {k}"
            at = caption_line.get(key)
            page = lambda idx: text[:sum(len(x) + 1 for x in lines[:idx])].count(chr(12)) + 1
            if at is None:
                rep.failure(f"caption of {key} not found in the PDF text")
            elif at > concl:
                rep.failure(f"{key} caption (page {page(at)}) comes after '6. Conclusion' (page {page(concl)}): float deferred past the conclusion")
            else:
                rep.out(f"ok    {key} caption on page {page(at)}, before the conclusion", always=True)
        for key, at in sorted(caption_line.items(), key=lambda kv: kv[1]):
            rep.out(f"      {key:10s} page {text[:sum(len(x) + 1 for x in lines[:at])].count(chr(12)) + 1}")
    for pat, label in ((r"\[verify", "'[verify'"), (r"\bpending\b", "'pending'"), (r"\bSWE-\d", "'SWE-<n>' agent label")):
        hits = [(text[:m.start()].count("\f") + 1, text[max(0, m.start() - 30):m.end() + 30].replace("\n", " ")) for m in re.finditer(pat, text)]
        if hits:
            rep.failure(f"{label} in the PDF: " + "; ".join(f"p{p}: …{c}…" for p, c in hits[:5]))
        else:
            rep.out(f"ok    no {label}", always=True)
    if len(pages) >= 2:
        header = next((l for l in pages[1].splitlines() if l.strip()), "")
        h = re.sub(r"\s+", "", header.lower())
        if "hokindeng.com" in h and "github.com" not in h:
            rep.out(f"ok    page-2 running header: {header.strip()[:90]}", always=True)
        else:
            rep.failure(f"page-2 running header is '{header.strip()[:90]}' (want hokindeng.com, not github.com)")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pdf", action="store_true", help="also inspect paper/main.pdf with pdftotext")
    ap.add_argument("--quiet", action="store_true", help="print only failures, warnings and summaries")
    ap.add_argument("--strict", action="store_true", help="a weakly resolved number (no named system/task in its sentence) is a failure")
    a = ap.parse_args()
    rep = Report(quiet=a.quiet)
    src = Sources()
    rep.out("== sources ==", always=True)
    rep.out(f"table1 {len(src.t1)} systems | table2 {len(src.t2)} agents | table3 {len(src.t3)} lanes | table4 {len(src.t4)} tasks | stats.json {len(src.stats)} lanes", always=True)
    for n in src.notes:
        rep.out(n, always=True)
        if "MISMATCH" in n:
            rep.fail += 1
    check_numbers(src, rep, strict=a.strict)
    check_abstract(src, rep)
    check_counts(src, rep)
    rep.out("\n== (d) generated tables recomputed from the CSVs ==", always=True)
    check_tab_main(src, rep)
    check_tab_categories(src, rep)
    check_tab_efficiency(src, rep)
    check_tab_hardest(src, rep)
    if a.pdf:
        check_pdf(rep)
    print(f"\n== summary: {rep.fail} failure(s), {rep.warn} warning(s) ==")
    sys.exit(1 if rep.fail else 0)


if __name__ == "__main__":
    main()
