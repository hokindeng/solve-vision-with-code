#!/usr/bin/env python3
"""Gate for solve-vision-with-code.com (the site is this docs/ folder). Nothing is pushed until this exits 0.

    python3 docs/verify-site.py                                   # the local docs/ tree
    python3 docs/verify-site.py --live https://solve-vision-with-code.com
    python3 docs/verify-site.py --live-only https://solve-vision-with-code.com

Checks (stdlib only; exit 1 on any FAIL, WARN never fails):
  banned strings  over every served text file (html css js json xml txt svg md py) and the paper's text (pdftotext when
                  present): organisation mailbox / repository prefix, bucket URLs, instance ids, IPv4 addresses, cloud
                  host names, plus the private patterns from ~/hokin/tools/site-banned-patterns.txt when that file
                  exists (this file is itself served, so it never carries them). Published model and lane names are
                  data and stay (DESIGN.md §0).
  links           every relative href / src / css url() in the pages resolves under docs/ (HEAD 200 in --live)
  data            tasks.json: 100 tasks, 50 In-Domain + 50 Out-of-Domain, 5 instances each, three closed lanes in the
                  panel order, every video / frame / program path exists, the 1,500 programs are exactly the files
                  under data/solve; leaderboard.json: 37 rows, top overall 0.923, 22 category rows, 27 efficiency rows
  pages           index.html + leaderboard.html: canonical and og:url are the domain, cite url is the domain,
                  no nav bar, no unfilled placeholder, footer lines, GitHub Pages URL not used as canonical
  hygiene         no Co-Authored-By trailer in the last 5 commits touching docs/
  --live          TLS (https 200, http -> 301, www), cert expiry, the domain and GitHub Pages serve identical
                  index.html / leaderboard.html / app.js / leaderboard.js / style.css, both equal the local files,
                  200 random media + program files HEAD 200

build_site.py imports BANNED (a list of compiled regexes, neutral + private) from here via importlib.
"""
import argparse
import concurrent.futures as cf
import json
import os
import random
import re
import shutil
import socket
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DOMAIN = "solve-vision-with-code.com"
PAGES_URL = "https://hokindeng.github.io/solve-vision-with-code/"
CLOSED_LANES = ["codex", "claude", "gemini"]  # panel column order; ground truth last
TEXT_EXT = {".html", ".htm", ".css", ".js", ".json", ".xml", ".txt", ".svg", ".md", ".py", ".webmanifest"}
PATH_EXT = (".mp4", ".webm", ".jpg", ".jpeg", ".png", ".gif", ".py", ".json", ".pdf", ".svg", ".css", ".js", ".html")
PAGES = ["index.html", "leaderboard.html"]
EXTERNAL_OK = {"fonts.googleapis.com", "fonts.gstatic.com", "github.com", "hokindeng.com", "www.hokindeng.com",
               DOMAIN, "www." + DOMAIN, "arxiv.org", "doi.org", "hokindeng.github.io", "huggingface.co"}

# Neutral patterns live here. The sensitive ones (organisation name, mail domain, org repository) are NOT written
# into this file — it is served from docs/ and the repo is public — they are read from an external file when present:
#   ~/hokin/tools/site-banned-patterns.txt   one regex per line, "#" comments; missing file -> neutral list only (WARN).
BANNED_FILE = os.environ.get("SITE_BANNED_PATTERNS", os.path.expanduser("~/hokin/tools/site-banned-patterns.txt"))
BANNED_RULES = [
    ("organisation mailbox", re.compile(r"inquiry@", re.I)),
    ("organisation repository prefix", re.compile(r"\bhipo-", re.I)),
    ("bucket URL", re.compile(r"s3://")),
    ("instance id", re.compile(r"\bi-0[0-9a-f]{8,}\b")),
    ("dotted IPv4", re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")),
    ("cloud host name", re.compile(r"amazonaws", re.I)),
]
EXTERNAL_LOADED = False
if os.path.isfile(BANNED_FILE):
    with open(BANNED_FILE, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#"):
                BANNED_RULES.append(("private pattern", re.compile(_line, re.I)))
                EXTERNAL_LOADED = True
BANNED = [rx for _label, rx in BANNED_RULES]  # what build_site.py imports
PLACEHOLDER = re.compile(r"__[A-Z][A-Z0-9_]{2,}__")
ALLOW_NAV = False


def banned_hits(text):
    """[(label, matched_text)] for every banned pattern in text."""
    return [(label, m.group(0)) for label, rx in BANNED_RULES for m in rx.finditer(text)]


# ----------------------------------------------------------------------------- report
class Report:
    def __init__(self):
        self.fails, self.warns, self.ok = [], [], 0

    def check(self, cond, msg):
        if cond:
            self.ok += 1
            print(f"  OK    {msg}")
        else:
            self.fails.append(msg)
            print(f"  FAIL  {msg}")
        return bool(cond)

    def warn(self, msg):
        self.warns.append(msg)
        print(f"  WARN  {msg}")

    def info(self, msg):
        print(f"  ..    {msg}")

    def section(self, title):
        print(f"\n== {title}")

    def finish(self):
        print(f"\n{self.ok} ok, {len(self.warns)} warn, {len(self.fails)} fail")
        for m in self.fails:
            print(f"  FAIL  {m}")
        return 1 if self.fails else 0


R = Report()


# ----------------------------------------------------------------------------- sources
class LocalTree:
    def __init__(self, root):
        self.root, self.label = root, root

    def exists(self, rel):
        return os.path.isfile(os.path.join(self.root, rel))

    def read_bytes(self, rel):
        with open(os.path.join(self.root, rel), "rb") as f:
            return f.read()

    def text(self, rel):
        return self.read_bytes(rel).decode("utf-8", "replace") if self.exists(rel) else None

    def walk(self):
        for dp, dn, fn in os.walk(self.root):
            dn[:] = [d for d in dn if d not in ("__pycache__", ".git")]
            for n in fn:
                yield os.path.relpath(os.path.join(dp, n), self.root).replace(os.sep, "/")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


class LiveSite:
    def __init__(self, base, timeout=20):
        self.base, self.label, self.timeout = base.rstrip("/"), base.rstrip("/"), timeout
        self._head = {}
        self.opener = urllib.request.build_opener()
        self.opener_nofollow = urllib.request.build_opener(NoRedirect)

    def url(self, rel):
        return rel if rel.startswith("http") else self.base + "/" + rel.lstrip("/")

    def fetch(self, rel, method="GET", follow=True):
        req = urllib.request.Request(self.url(rel), method=method, headers={"User-Agent": "verify-site/1"})
        try:
            with (self.opener if follow else self.opener_nofollow).open(req, timeout=self.timeout) as r:
                return r.status, dict(r.headers), (r.read() if method == "GET" else b"")
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers), b""
        except Exception as e:
            return 0, {"error": repr(e)}, b""

    def head(self, rel):
        if rel not in self._head:
            st, hd, _ = self.fetch(rel, "HEAD")
            if st == 405:
                st, hd, _ = self.fetch(rel, "GET")
            self._head[rel] = (st, hd)
        return self._head[rel]

    def exists(self, rel):
        return self.head(rel)[0] == 200

    def read_bytes(self, rel):
        st, _, body = self.fetch(rel)
        return body if st == 200 else None

    def text(self, rel):
        b = self.read_bytes(rel)
        return None if b is None else b.decode("utf-8", "replace")


# ----------------------------------------------------------------------------- helpers
def scan(name, text):
    by = {}
    for label, m in banned_hits(text):
        by.setdefault(label, []).append(m)
    for label, ms in by.items():
        R.check(False, f"{name}: {label} x{len(ms)}  e.g. {ms[0]!r}")
    return sum(len(v) for v in by.values())


def pdf_text(data):
    if not shutil.which("pdftotext"):
        return None
    tmp = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"verify-site-{os.getpid()}.pdf")
    with open(tmp, "wb") as f:
        f.write(data)
    try:
        return subprocess.run(["pdftotext", "-q", tmp, "-"], capture_output=True, text=True, timeout=120).stdout
    except Exception as e:
        R.warn(f"pdftotext failed: {e}")
        return None
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


SCRIPT_RX = re.compile(r"<script\b([^>]*)>(.*?)</script>", re.S | re.I)
ATTR_RX = re.compile(r"""(?:href|src|poster|data-src)\s*=\s*["']([^"']+)["']""", re.I)
CSS_URL_RX = re.compile(r"""url\(\s*["']?([^"')]+?)["']?\s*\)""", re.I)
TAG_RX = re.compile(r"<(link|meta)\b([^>]*)>", re.I)
TAG_ATTR_RX = re.compile(r"""([\w:-]+)\s*=\s*["']([^"']*)["']""")


def tag_attrs(html, tag):
    for m in TAG_RX.finditer(html):
        if m.group(1).lower() == tag:
            yield dict((k.lower(), v) for k, v in TAG_ATTR_RX.findall(m.group(2)))


def canonical_of(html):
    return next((a.get("href") for a in tag_attrs(html, "link") if a.get("rel", "").lower() == "canonical"), None)


def og_url_of(html):
    return next((a.get("content") for a in tag_attrs(html, "meta") if a.get("property", "").lower() == "og:url"), None)


def internal_refs(text, kind):
    """(raw, path-or-None) for every reference; None path = external."""
    if kind == "html":
        markup = SCRIPT_RX.sub("", re.sub(r"<!--.*?-->", "", text, flags=re.S))
        refs = ATTR_RX.findall(markup) + CSS_URL_RX.findall(markup)
    else:
        refs = CSS_URL_RX.findall(text)
    for raw in refs:
        r = raw.strip()
        if not r or r.startswith(("#", "mailto:", "tel:", "data:", "javascript:", "about:")):
            continue
        if r.startswith(("http://", "https://", "//")):
            yield raw, None
        else:
            yield raw, urllib.parse.urlsplit(r).path


def resolve_rel(page_rel, path):
    if path.startswith("/"):
        p = path.lstrip("/")  # DESIGN.md: paths stay relative; a leading slash breaks GitHub Pages — flagged below
    else:
        p = os.path.normpath(os.path.join(os.path.dirname(page_rel), path)).replace(os.sep, "/")
        if p == ".":
            p = ""
    if p.endswith("/") or p == "":
        p += "index.html"
    return p


def walk_paths(obj, out):
    """Every string in a JSON document that looks like a relative media / data path."""
    if isinstance(obj, dict):
        for v in obj.values():
            walk_paths(v, out)
    elif isinstance(obj, list):
        for v in obj:
            walk_paths(v, out)
    elif isinstance(obj, str) and obj.startswith(("media/", "data/")) and obj.endswith(PATH_EXT):
        out.append(obj)


# ----------------------------------------------------------------------------- checks
def check_pages(src):
    refs_by_page = {}
    for page in PAGES:
        html = src.text(page)
        R.section(f"page {page}")
        if not R.check(html is not None, f"{page} present"):
            continue
        want = f"https://{DOMAIN}/" + ("" if page == "index.html" else page)
        can, og = canonical_of(html), og_url_of(html)
        R.check(can == want, f"{page}: canonical == {want} (got {can!r})")
        R.check(og == want, f"{page}: og:url == {want} (got {og!r})")
        R.check(bool(re.search(r"<title>[^<]{3,}</title>", html, re.I)), f"{page}: <title>")
        if "<nav" in html.lower() and ALLOW_NAV:
            R.warn(f"{page}: nav bar markup present (allowed by --allow-nav)")
        else:
            R.check("<nav" not in html.lower(), f"{page}: no nav bar markup")
        ph = sorted(set(PLACEHOLDER.findall(html)))
        R.check(not ph, f"{page}: no unfilled placeholder (got {ph})")
        R.check("hokindeng.com" in html and "github.com/hokindeng/solve-vision-with-code" in html,
                f"{page}: footer carries hokindeng.com and the GitHub link")
        R.check(PAGES_URL not in (can or "") and PAGES_URL not in (og or ""), f"{page}: GitHub Pages URL is not the canonical")
        if 'name="viewport"' not in html:
            R.warn(f"{page}: no viewport meta")
        if page == "index.html":
            cite = re.search(r"url\s*=\s*\{([^}]*)\}", html)
            R.check(bool(cite) and cite.group(1).strip() == f"https://{DOMAIN}/",
                    f"index.html: cite url == https://{DOMAIN}/ (got {cite.group(1).strip() if cite else None!r})")
            for word in ("Paper", "Code", "Benchmark", "Leaderboard", "Cite"):
                if not re.search(r">\s*" + word + r"\s*<", html):
                    R.warn(f"index.html: hero button {word!r} not found")
            R.check("paper.pdf" in html, "index.html links paper.pdf")
        refs = list(internal_refs(html, "html"))
        refs_by_page[page] = refs
        leading = [raw for raw, p in refs if p and raw.strip().startswith("/")]
        R.check(not leading, f"{page}: no root-relative paths (GitHub Pages serves under /solve-vision-with-code/) — {leading[:3]}")
        # the page's own css / js get scanned and their url()s followed
        for raw, p in refs:
            if p and p.endswith(".css"):
                t = src.text(resolve_rel(page, p))
                if t is not None and p not in refs_by_page:
                    refs_by_page[p] = list(internal_refs(t, "css"))
    return refs_by_page


def check_links(src, refs_by_page):
    R.section(f"links on {src.label}")
    missing, external, seen = [], set(), set()
    for page, refs in refs_by_page.items():
        for raw, path in refs:
            if path is None:
                host = urllib.parse.urlsplit(raw if not raw.startswith("//") else "https:" + raw).hostname or ""
                if host not in EXTERNAL_OK:
                    external.add(host)
                continue
            target = resolve_rel(page, path)
            if target in seen:
                continue
            seen.add(target)
            if not src.exists(target):
                missing.append(f"{page} -> {raw}")
    R.check(not missing, f"{len(seen)} internal references resolve ({len(missing)} missing: {missing[:5]})")
    if external:
        R.warn(f"external hosts outside the allow-list: {sorted(external)}")


def check_data(src):
    R.section("data/tasks.json")
    raw = src.text("data/tasks.json")
    if not R.check(raw is not None, "data/tasks.json present"):
        return []
    t = json.loads(raw)
    tasks = t.get("tasks") or []
    R.check(len(tasks) == 100, f"100 tasks (got {len(tasks)})")
    R.check(t.get("closed_lanes") == CLOSED_LANES, f"closed_lanes == {CLOSED_LANES} (got {t.get('closed_lanes')})")
    R.check(len(t.get("open_lanes") or []) == 3, f"3 open lanes (got {len(t.get('open_lanes') or [])})")
    lanes = t.get("lanes") or {}
    R.check(all(l in lanes and lanes[l].get("label") for l in CLOSED_LANES), "every closed lane has a label")
    splits = {}
    for x in tasks:
        splits[x.get("split")] = splits.get(x.get("split"), 0) + 1
    R.check(splits == {"In-Domain_50": 50, "Out-of-Domain_50": 50}, f"50 In-Domain + 50 Out-of-Domain (got {splits})")
    ids = [x.get("id") for x in tasks]
    R.check(len(set(ids)) == 100, "task ids unique")
    R.check(all(x.get("category") and x.get("short") and x.get("prompt") for x in tasks), "every task has category, short, prompt")
    n_inst = sum(len(x.get("instances") or []) for x in tasks)
    R.check(n_inst == 500 and all(len(x.get("instances") or []) == 5 for x in tasks), f"500 instances, 5 per task (got {n_inst})")
    bad_lane, bad_score, solves = [], [], []
    for x in tasks:
        for inst in x.get("instances") or []:
            ln = inst.get("lanes") or {}
            for l in CLOSED_LANES:
                e = ln.get(l)
                if not e or not e.get("video") or not e.get("solve_url"):
                    bad_lane.append(f"{x.get('short')}/{inst.get('idx')}/{l}")
                    continue
                solves.append(e["solve_url"])
                s = e.get("score")
                if not isinstance(s, (int, float)) or not 0 <= s <= 1:
                    bad_score.append(f"{x.get('short')}/{inst.get('idx')}/{l}={s!r}")
    R.check(not bad_lane, f"every instance has video + program for the 3 closed lanes ({len(bad_lane)} missing: {bad_lane[:3]})")
    R.check(not bad_score, f"every closed-lane score is a number in [0,1] ({len(bad_score)} bad: {bad_score[:3]})")
    R.check(len(solves) == 1500 and len(set(solves)) == 1500, f"1,500 distinct programs referenced (got {len(set(solves))})")
    paths = []
    walk_paths(t, paths)
    paths = sorted(set(paths))
    missing = [p for p in paths if not src.exists(p)]
    R.check(not missing, f"all {len(paths):,} media / program paths in tasks.json exist on {src.label} ({len(missing)} missing: {missing[:3]})")
    if isinstance(src, LocalTree):
        on_disk = sorted(p for p in src.walk() if p.startswith("data/solve/") and p.endswith(".py"))
        R.check(len(on_disk) == 1500, f"1,500 program files under data/solve (got {len(on_disk)})")
        extra = sorted(set(on_disk) - set(solves))
        R.check(not extra, f"no program file outside tasks.json ({len(extra)} extra: {extra[:3]})")
        media = [p for p in src.walk() if p.startswith("media/")]
        orphans = sorted(set(media) - set(paths))
        if orphans:
            R.warn(f"{len(orphans)} media files not referenced by tasks.json: {orphans[:3]}")
        R.info(f"media/: {len(media):,} files; referenced: {len([p for p in paths if p.startswith('media/')]):,}")

    R.section("data/leaderboard.json")
    raw = src.text("data/leaderboard.json")
    if R.check(raw is not None, "data/leaderboard.json present"):
        lb = json.loads(raw)
        rows = lb.get("rows") or []
        R.check(len(rows) == 37, f"37 leaderboard rows (got {len(rows)})")
        R.check(len(lb.get("categories") or []) == 22 and len(lb.get("efficiency") or []) == 27,
                f"22 category rows, 27 efficiency rows (got {len(lb.get('categories') or [])}, {len(lb.get('efficiency') or [])})")
        top = max((r.get("overall") or 0 for r in rows), default=0)
        R.check(abs(top - 0.923) < 0.0005, f"top overall 0.923 (got {top:.4f})")
        agents = [r for r in rows if r.get("kind") == "coding agent"]
        videos = [r for r in rows if r.get("kind") == "video model"]
        if agents and videos:
            ba, bv = max(agents, key=lambda r: r["overall"]), max(videos, key=lambda r: r["overall"])
            R.info(f"hero numbers: best coding agent {ba['model']} {ba['overall']:.3f} · best video model {bv['model']} {bv['overall']:.3f}")
    return paths


def check_static(src):
    R.section(f"static files on {src.label}")
    pdf = src.read_bytes("paper.pdf") if src.exists("paper.pdf") else None
    R.check(pdf is not None and pdf[:4] == b"%PDF" and len(pdf) > 100_000, "paper.pdf present, real PDF, > 100 KB")
    if pdf:
        txt = pdf_text(pdf)
        if txt is None:
            R.warn("pdftotext not installed — paper text not scanned")
        else:
            scan("paper.pdf(text)", txt)
            R.info(f"paper.pdf text scanned ({len(txt):,} chars)")
    for f in ("style.css", "app.js", "leaderboard.js"):
        R.check(src.exists(f), f"{f} present")
    if isinstance(src, LocalTree) and not src.exists(".nojekyll"):
        R.warn(".nojekyll missing (GitHub Pages would run Jekyll over docs/)")


def check_banned_tree(tree):
    R.section(f"banned strings in {tree.root}")
    n = 0
    me = os.path.basename(__file__)
    for rel in tree.walk():
        if rel == me:
            continue  # this file holds the neutral regexes themselves
        if os.path.splitext(rel)[1].lower() in TEXT_EXT:
            n += 1
            scan(rel, tree.text(rel))
    R.info(f"{n:,} text files scanned against {len(BANNED_RULES)} patterns")
    R.check(n > 0, "text files found to scan")
    if EXTERNAL_LOADED:
        R.info(f"private patterns loaded from {BANNED_FILE}")
    else:
        R.warn(f"{BANNED_FILE} not found — only the neutral patterns were checked (run from the harness for the full list)")


def check_git():
    R.section("git hygiene (public repo)")
    try:
        out = subprocess.run(["git", "-C", REPO, "log", "-5", "--format=%H%n%B%x00", "--", "docs"],
                             capture_output=True, text=True, timeout=30).stdout
    except Exception as e:
        R.warn(f"git not available: {e}")
        return
    bad = [c.splitlines()[0][:10] for c in out.split("\x00") if "co-authored-by" in c.lower()]
    R.check(not bad, f"no Co-Authored-By trailer in the last 5 commits touching docs/ (bad: {bad})")


def check_live(live, local, paths, sample_n, seed):
    R.section(f"live {live.base}")
    host = urllib.parse.urlsplit(live.base).hostname
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((host, 443), timeout=15) as s, ctx.wrap_socket(s, server_hostname=host) as ss:
            cert = ss.getpeercert()
        days = (ssl.cert_time_to_seconds(cert["notAfter"]) - time.time()) / 86400
        R.check(days > 0, f"TLS certificate valid for {host} ({days:.0f} days left)")
        if days < 21:
            R.warn(f"certificate expires in {days:.0f} days")
    except Exception as e:
        R.check(False, f"TLS handshake with {host}: {e!r}")
    st, hd, _ = live.fetch(f"http://{host}/", follow=False)
    R.check(st == 301 and hd.get("Location", "").startswith("https://"), f"http://{host}/ -> 301 https (got {st} {hd.get('Location')})")
    st, hd, _ = live.fetch(f"https://www.{host}/", follow=False)
    R.check(st in (200, 301), f"https://www.{host}/ answers (got {st} {hd.get('Location', '')})")

    for rel in ("paper.pdf", "data/tasks.json", "data/leaderboard.json"):
        st, _ = live.head(rel)
        R.check(st == 200, f"HEAD /{rel} -> {st}")

    pages = LiveSite(PAGES_URL)
    compare = PAGES + ["app.js", "leaderboard.js", "style.css"]
    live_bodies = {}
    for rel in compare:
        d, p = live.read_bytes(rel), pages.read_bytes(rel)
        R.check(d is not None, f"domain serves /{rel}")
        R.check(p is not None, f"GitHub Pages serves {rel}")
        if d is not None and p is not None:
            R.check(d == p, f"domain and GitHub Pages serve identical {rel}" + ("" if d == p else " — box not pulled yet, or Pages still building"))
        if d is not None and local is not None:
            R.check(local.read_bytes(rel) == d, f"live {rel} == local docs/{rel}" + ("" if local.read_bytes(rel) == d else " — deploy not landed"))
        if d is not None:
            live_bodies[rel] = d

    refs_by_page = {}
    for rel, body in live_bodies.items():
        text = body.decode("utf-8", "replace")
        scan(f"live {rel}", text)
        if rel.endswith(".html"):
            refs_by_page[rel] = list(internal_refs(text, "html"))
        elif rel.endswith(".css"):
            refs_by_page[rel] = list(internal_refs(text, "css"))
    check_links(live, refs_by_page)

    if not paths:
        raw = live.text("data/tasks.json")
        if raw:
            walk_paths(json.loads(raw), paths)
    paths = sorted(set(paths))
    if paths:
        sample = random.Random(seed).sample(paths, min(sample_n, len(paths)))
        with cf.ThreadPoolExecutor(8) as ex:
            res = list(ex.map(lambda r: (r, live.head(r)), sample))
        bad = [f"{r} -> {st}" for r, (st, _h) in res if st != 200]
        R.check(not bad, f"{len(sample)} random media / program files HEAD 200 on the domain ({len(bad)} bad: {bad[:5]})")
    else:
        R.warn("no media references to sample")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=HERE, help="the docs/ tree (default: this file's folder)")
    ap.add_argument("--live", metavar="URL", help="also verify the deployed site")
    ap.add_argument("--live-only", metavar="URL", help="verify only the deployed site")
    ap.add_argument("--allow-nav", action="store_true", help="do not fail on <nav> markup")
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260921)
    a = ap.parse_args()
    global ALLOW_NAV
    ALLOW_NAV = a.allow_nav

    local, paths = None, []
    if not a.live_only:
        local = LocalTree(a.root)
        R.section(f"local tree {a.root}")
        if not R.check(os.path.isdir(a.root), f"docs root exists: {a.root}"):
            return R.finish()
        refs_by_page = check_pages(local)
        check_links(local, refs_by_page)
        paths = check_data(local)
        check_static(local)
        check_banned_tree(local)
        check_git()
    url = a.live or a.live_only
    if url:
        check_live(LiveSite(url), local, paths, a.sample, a.seed)
    return R.finish()


if __name__ == "__main__":
    sys.exit(main())
