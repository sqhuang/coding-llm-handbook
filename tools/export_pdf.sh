#!/usr/bin/env bash
# Export the rendered index.html → a single PDF using Chrome headless.
# Uses the existing print stylesheet baked into the site.
#
# REQUIRES: macOS Chrome / Chromium / Edge (any chromium-family browser).
# OUTPUT  : ./cookbook.pdf
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -f index.html ]]; then
  echo "index.html not found — run \`python build.py\` first" >&2
  exit 1
fi

# Locate a chromium binary
CHROME=""
for candidate in \
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  "/Applications/Chromium.app/Contents/MacOS/Chromium" \
  "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge" \
  "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser" \
  "$(command -v google-chrome 2>/dev/null || true)" \
  "$(command -v chromium 2>/dev/null || true)" \
  "$(command -v chrome 2>/dev/null || true)"
do
  if [[ -n "$candidate" && -x "$candidate" ]]; then
    CHROME="$candidate"
    break
  fi
done

if [[ -z "$CHROME" ]]; then
  echo "No Chromium-family browser found." >&2
  echo "Install Chrome / Edge / Brave / Chromium, or override CHROME=<path>." >&2
  exit 2
fi

OUT=${1:-cookbook.pdf}
URL="file://$(pwd)/index.html"

echo "→ Exporting $URL → $OUT (using $(basename "$CHROME"))"
"$CHROME" \
  --headless=new \
  --disable-gpu \
  --no-sandbox \
  --no-pdf-header-footer \
  --print-to-pdf-no-header \
  --print-to-pdf="$OUT" \
  --virtual-time-budget=30000 \
  --run-all-compositor-stages-before-draw \
  "$URL"

if [[ -f "$OUT" ]]; then
  size=$(du -h "$OUT" | cut -f1)
  echo "✓ Wrote $OUT ($size)"
else
  echo "✗ PDF export failed" >&2
  exit 3
fi

echo
echo "Note · This is a single-page-flow PDF using the site's @media print rules."
echo "      For per-chapter PDFs, open each chapter in the live site and use"
echo "      browser File → Print → Save as PDF."
