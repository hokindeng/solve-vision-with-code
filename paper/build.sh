#!/usr/bin/env bash
# Build paper/main.pdf: pdflatex → bibtex → pdflatex → pdflatex.
# Exit non-zero on any LaTeX error or a missing PDF. Afterwards print
# undefined references / citations and overfull boxes wider than 20 pt.
#
#   bash paper/build.sh            # full build
#   bash paper/build.sh --check    # toolchain check only (binaries + every package main.tex loads)
#
# BasicTeX note: packages missing from /usr/local/texlive/2025basic live in ~/Library/texmf
# (titlesec, enumitem, needspace, lastpage were installed there by hand); never install with sudo.

set -u
cd "$(dirname "$0")"
export PATH="/Library/TeX/texbin:$PATH"

MAIN=main
LOG="$MAIN.log"
OVERFULL_PT=20

red()   { printf '\033[31m%s\033[0m\n' "$*"; }
green() { printf '\033[32m%s\033[0m\n' "$*"; }

# ---------- toolchain check ----------
check_toolchain() {
  local ok=0
  for bin in pdflatex bibtex kpsewhich; do
    if ! command -v "$bin" >/dev/null 2>&1; then red "missing binary: $bin"; ok=1; fi
  done
  # every \usepackage{...} in main.tex (comment lines skipped; comma lists split)
  local pkgs
  pkgs=$(grep -v '^\s*%' "$MAIN.tex" | grep -o '\\usepackage\(\[[^]]*\]\)\?{[^}]*}' \
         | sed -E 's/.*\{([^}]*)\}/\1/' | tr ',' '\n' | sed 's/ //g' | sort -u)
  for p in $pkgs; do
    if [ -f "$p.sty" ]; then continue; fi                 # shipped next to main.tex (icml2026, algorithm*)
    if ! kpsewhich "$p.sty" >/dev/null 2>&1; then red "missing package: $p.sty"; ok=1; fi
  done
  for f in icml2026.sty icml2026.bst; do
    [ -f "$f" ] || { red "missing local style file: $f"; ok=1; }
  done
  return $ok
}

if [ "${1:-}" = "--check" ]; then
  check_toolchain && green "toolchain OK" ; exit $?
fi
check_toolchain || { red "toolchain check failed"; exit 2; }

# ---------- build ----------
run_latex() {
  pdflatex -interaction=nonstopmode -halt-on-error -file-line-error "$MAIN.tex" >/dev/null 2>&1
  return $?
}

report_errors() {
  # file:line: message lines from -file-line-error, plus classic "! " lines
  grep -E '^(\./)?[^ ]+\.(tex|sty|bst|bbl):[0-9]+: |^! ' "$LOG" | head -40
}

status=0
run_latex || status=$?
if [ $status -ne 0 ]; then
  red "pdflatex pass 1 failed"; report_errors; exit 1
fi

if grep -q '\\bibdata' "$MAIN.aux" 2>/dev/null; then
  bibtex "$MAIN" >"$MAIN.blg.out" 2>&1 || {
    red "bibtex reported errors:"; grep -E "^(I couldn't|I found no|Warning--|error)" "$MAIN.blg" | head -20
    # bibtex exit 2 = errors (missing database entry etc.); continue so the pdf still builds
  }
  rm -f "$MAIN.blg.out"
else
  echo "no \\bibdata in aux (main.bib missing or no \\bibliography); bibtex skipped"
fi

run_latex || { red "pdflatex pass 2 failed"; report_errors; exit 1; }
run_latex || { red "pdflatex pass 3 failed"; report_errors; exit 1; }

[ -f "$MAIN.pdf" ] || { red "no $MAIN.pdf produced"; exit 1; }

# ---------- diagnostics ----------
pages=$(grep -o 'Output written on [^ ]* ([0-9]* pages' "$LOG" | grep -o '[0-9]* pages')
green "built $MAIN.pdf ($pages)"

echo "--- undefined references"
grep -o "Reference \`[^']*' on page [0-9]* undefined" "$LOG" | sort -u || true
echo "--- undefined citations"
grep -o "Citation \`[^']*' on page [0-9]* undefined" "$LOG" | sed -E "s/ on page [0-9]+//" | sort -u || true
echo "--- multiply-defined labels"
grep -o "Label \`[^']*' multiply defined" "$LOG" | sort -u || true
echo "--- missing files (figures / inputs)"
grep -E "LaTeX Warning: File \`[^']*' not found|! LaTeX Error: File \`[^']*' not found" "$LOG" | sort -u || true
echo "--- overfull boxes > ${OVERFULL_PT}pt"
grep -E '^Overfull \\[hv]box \([0-9.]+pt too' "$LOG" | awk -v lim="$OVERFULL_PT" '{
     s=$0; sub(/^[^(]*\(/, "", s); sub(/pt.*/, "", s); if (s+0 > lim) print $0 }' || true
echo "--- pending markers in the PDF text"
if command -v pdftotext >/dev/null 2>&1; then
  pdftotext "$MAIN.pdf" - 2>/dev/null | grep -c '^\[.*\]$' | sed 's/^/lines that are only a [..] marker: /'
fi
exit 0
