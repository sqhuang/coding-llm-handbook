#!/usr/bin/env python3
"""
Build a single self-contained index.html.

- Pre-renders Mermaid blocks to SVG (light + dark) via mmdc, with on-disk caching
- Splits HTML / CSS / JS into assets/ for maintainability
- Computes per-chapter reading time
"""
import concurrent.futures
import datetime
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
ASSETS = HERE / "assets"
CACHE = HERE / "cache" / "mermaid"
CACHE.mkdir(parents=True, exist_ok=True)

NAV = [
    ("readme",     "总览",     "〇", "总览 · Overview"),
    ("roadmap",    "路线图",   "◎", "九阶路线图"),
    ("basics",     "基础",     "·", "训练基础 · 从 CV 到文本训练"),
    ("phase0",     "全景",     "0", "GLM-5.1 架构 & 全景对比"),
    ("phase1",     "数据",     "1", "预训练数据 pipeline"),
    ("phase2",     "架构",     "2", "预训练架构与实操"),
    ("phase3",     "长上下文", "3", "中期训练 & 长上下文"),
    ("phase4",     "微调",     "4", "SFT 与 Agent 轨迹"),
    ("phase5",     "强化学习", "5", "RL · RLHF / RLVR / Agentic"),
    ("phase6",     "评测",     "6", "评测体系"),
    ("phase7",     "部署",     "7", "推理部署优化"),
    ("phase8",     "应用",     "8", "Coding Agent 应用"),
    ("tooluse",    "Tool Use", "⚒", "Tool Use 速读 · 横切主线 + 业界 8 例"),
    ("lab",        "实验",     "✦", "实验册 · 30 分钟可跑的 A vs B 对比"),
    ("capstone",   "Capstone", "✪", "Capstone · 4 周端到端实验 + 看板"),
    ("outro",      "结语",     "★", "结语 · 从读完到真正上手"),
    ("glossary",   "索引",     "▣", "概念索引 · Glossary"),
    ("references", "参考",     "❉", "参考文献"),
]

FILES = {
    "readme":     "README.md",
    "roadmap":    "ROADMAP.md",
    "basics":     "phase_basics_training.md",
    "phase0":     "phase0_foundation.md",
    "phase1":     "phase1_data_pipeline.md",
    "phase2":     "phase2_pretraining.md",
    "phase3":     "phase3_midtraining_longcontext.md",
    "phase4":     "phase4_sft.md",
    "phase5":     "phase5_rl.md",
    "phase6":     "phase6_evaluation.md",
    "phase7":     "phase7_deployment.md",
    "phase8":     "phase8_agent_apps.md",
    "tooluse":    "phase_tooluse.md",
    "lab":        "phase_lab.md",
    "capstone":   "phase_capstone.md",
    "outro":      "phase_outro.md",
    "glossary":   "phase_glossary.md",
    "references": "phase_references.md",
}

MMDC = HERE / "node_modules" / ".bin" / "mmdc"
CFG_LIGHT = ASSETS / "mermaid-light.json"
CFG_DARK = ASSETS / "mermaid-dark.json"

# Match a ```mermaid ... ``` fenced block (multiline)
MERMAID_RE = re.compile(r'^```mermaid\s*\n([\s\S]+?)^```\s*$', re.MULTILINE)

# Match `<!-- include: <path> [as <lang>] -->` on its own line.
# Path is relative to repo root. Optional `as` overrides the fence language
# (default = file extension).
INCLUDE_RE = re.compile(
    r'^[ \t]*<!--\s*include:\s*([^\s]+?)(?:\s+as\s+([A-Za-z0-9_+\-]+))?\s*-->[ \t]*$',
    re.MULTILINE,
)

# Reverse map: source markdown filename -> nav id. Used to rewrite cross-file
# md links into SPA anchors so they resolve in the rendered site. The GitHub
# README still uses the .md paths (untouched source on disk).
MD_TO_NAVID = {v: k for k, v in FILES.items()}
NAV_IDS = set(FILES.keys())


def render_mermaid(source: str, theme: str) -> tuple[str, bool]:
    """Render mermaid source to SVG string, cached on disk by content hash.

    Returns (svg_string, ok_flag).
    """
    cfg = CFG_LIGHT if theme == "light" else CFG_DARK
    h = hashlib.sha256((source + theme).encode("utf-8")).hexdigest()[:16]
    out = CACHE / f"{h}-{theme}.svg"
    if out.exists():
        return out.read_text(encoding="utf-8"), True

    inp = CACHE / f"{h}.mmd"
    inp.write_text(source, encoding="utf-8")
    cmd = [
        str(MMDC),
        "-i", str(inp),
        "-o", str(out),
        "-c", str(cfg),
        "-b", "transparent",
        "--quiet",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not out.exists():
        err = (result.stderr or result.stdout or "").strip().splitlines()
        err_short = " | ".join(err[-3:]) if err else "unknown error"
        print(f"  ! mermaid render failed ({theme}): {err_short}")
        # Build a visible fallback inline SVG with the error text
        msg = f"Mermaid render failed ({theme}): {err_short}"
        msg = msg.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        fallback = (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 80">'
            f'<rect width="800" height="80" fill="#fff5e6" stroke="#a52619"/>'
            f'<text x="20" y="32" fill="#a52619" font-family="monospace" font-size="13">'
            f'⚠ {msg[:90]}</text>'
            f'<text x="20" y="56" fill="#a52619" font-family="monospace" font-size="11">'
            f'(see source for diagram)</text>'
            f'</svg>'
        )
        return fallback, False
    return out.read_text(encoding="utf-8"), True


_SVG_CLASS_RE = re.compile(r'<svg\b([^>]*)>')
_VIEWBOX_RE = re.compile(r'viewBox="\s*[\-0-9.]+\s+[\-0-9.]+\s+([0-9.]+)\s+([0-9.]+)"')


def aspect_class(svg: str) -> str:
    """Return an aspect-* class based on viewBox W/H ratio."""
    m = _VIEWBOX_RE.search(svg)
    if not m:
        return "aspect-normal"
    w, h = float(m.group(1)), float(m.group(2))
    if h <= 0:
        return "aspect-normal"
    r = w / h
    if r >= 7.0:
        return "aspect-very-wide"   # let overflow horizontally
    if r >= 2.0:
        return "aspect-wide"        # full width, normal scaling
    if r <= 0.4:
        return "aspect-very-tall"   # cap height
    if r <= 0.7:
        return "aspect-tall"        # softer height cap
    return "aspect-normal"


def post_process_svg(svg: str, css_class: str) -> str:
    """Strip explicit width/height, add CSS classes (theme + aspect) to root <svg>."""
    classes = f"{css_class} {aspect_class(svg)}"

    def fix_root(match):
        attrs = match.group(1)
        # Drop explicit width / height attributes (let CSS control)
        attrs = re.sub(r'\s(?:width|height)="[^"]*"', '', attrs)
        # Drop inline style that might cap max-width
        attrs = re.sub(r'\sstyle="[^"]*"', '', attrs)
        # Add or extend class
        if 'class="' in attrs:
            attrs = re.sub(r'class="([^"]*)"', f'class="\\1 {classes}"', attrs, count=1)
        else:
            attrs += f' class="{classes}"'
        return f'<svg{attrs}>'
    return _SVG_CLASS_RE.sub(fix_root, svg, count=1)


def _mermaid_cache_path(source: str, theme: str) -> Path:
    h = hashlib.sha256((source + theme).encode("utf-8")).hexdigest()[:16]
    return CACHE / f"{h}-{theme}.svg"


def prewarm_mermaid_cache(all_md: list[str], max_workers: int = 8) -> None:
    """Render any uncached (block, theme) pairs across all chapters in parallel.

    `replace_mermaid_blocks` is still called per-chapter afterwards, but every
    `render_mermaid` invocation will now hit the disk cache and return instantly.
    """
    misses: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for md in all_md:
        for m in MERMAID_RE.finditer(md):
            source = m.group(1).rstrip()
            for theme in ("light", "dark"):
                key = (source, theme)
                if key in seen:
                    continue
                seen.add(key)
                if not _mermaid_cache_path(source, theme).exists():
                    misses.append(key)
    if not misses:
        return
    print(f"  · pre-rendering {len(misses)} uncached mermaid svg(s) in parallel")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for _ in ex.map(lambda args: render_mermaid(*args), misses):
            pass


def replace_mermaid_blocks(md: str) -> tuple[str, int, int]:
    """Replace ```mermaid blocks with pre-rendered <div class="mermaid-block"> HTML.
    Returns (new_md, total_blocks, failed_blocks).
    """
    total = 0
    failed = 0

    def repl(m):
        nonlocal total, failed
        total += 1
        source = m.group(1).rstrip()
        light_svg, ok1 = render_mermaid(source, "light")
        dark_svg, ok2 = render_mermaid(source, "dark")
        if not (ok1 and ok2):
            failed += 1
        light = post_process_svg(light_svg, "mermaid-svg-light")
        dark = post_process_svg(dark_svg, "mermaid-svg-dark")
        # Inline raw HTML; surround with blank lines so marked treats as HTML block
        return f'\n<div class="mermaid-block">{light}{dark}</div>\n'
    new_md = MERMAID_RE.sub(repl, md)
    return new_md, total, failed


# ---------- Include directive: <!-- include: examples/foo.py [as python] -->

_EXT_TO_LANG = {
    "py": "python", "sh": "bash", "bash": "bash",
    "js": "javascript", "ts": "typescript",
    "yaml": "yaml", "yml": "yaml",
    "json": "json", "toml": "toml",
    "md": "markdown", "txt": "text",
    "rs": "rust", "go": "go", "c": "c", "cpp": "cpp", "h": "c",
    "cu": "cpp",
}


def process_includes(md: str) -> tuple[str, int, list[str]]:
    """Replace `<!-- include: path [as lang] -->` with a fenced code block
    holding the referenced file's contents.

    Paths resolve relative to the repo root (HERE). Missing files are kept as
    a visible error comment and reported back so the build can fail.
    """
    n = 0
    errors: list[str] = []

    def repl(m: re.Match) -> str:
        nonlocal n
        rel = m.group(1)
        lang_override = m.group(2)
        path = (HERE / rel).resolve()
        try:
            path.relative_to(HERE.resolve())
        except ValueError:
            errors.append(f"include path escapes repo root: {rel}")
            return f"<!-- include FAILED (path escapes repo): {rel} -->"
        if not path.exists():
            errors.append(f"include target not found: {rel}")
            return f"<!-- include FAILED (not found): {rel} -->"
        body = path.read_text(encoding="utf-8").rstrip()
        lang = lang_override or _EXT_TO_LANG.get(path.suffix.lstrip(".").lower(), "")
        n += 1
        # Caption line lets the reader jump to the source file on GitHub
        caption = f"<sub>📎 来自 [`{rel}`](./{rel})</sub>"
        return f"{caption}\n\n```{lang}\n{body}\n```"

    new_md = INCLUDE_RE.sub(repl, md)
    return new_md, n, errors


# ---------- Cross-md link rewrite

# Match `[label](./xxxx.md)` or `[label](./xxxx.md#frag)` references.
_LOCAL_MD_LINK_RE = re.compile(r'\]\(\./([A-Za-z0-9_.\-/]+\.md)(#[^)\s]+)?\)')


def rewrite_local_md_links(md: str) -> tuple[str, list[str]]:
    """Rewrite `](./phase_xxx.md)` into SPA-friendly `](#<navid>)`.

    Inputs that don't resolve to a known nav id are returned as warnings so the
    author sees the typo before shipping.
    """
    warnings: list[str] = []

    # Skip code blocks / inline code (same trick as add_xref_links)
    parts = re.split(r'(```[\s\S]*?```|`[^`\n]+`)', md)
    out = []
    for part in parts:
        if part.startswith('```') or (part.startswith('`') and not part.startswith('```')):
            out.append(part)
            continue

        def repl(m: re.Match) -> str:
            filename = m.group(1)
            frag = m.group(2) or ""
            nav_id = MD_TO_NAVID.get(filename)
            if not nav_id:
                warnings.append(f"unknown cross-md target: ./{filename}")
                return m.group(0)
            # Drop in-file fragments for now (SPA anchors are per-chapter h-N
            # ids and aren't author-addressable).
            return f"](#{nav_id})"

        out.append(_LOCAL_MD_LINK_RE.sub(repl, part))
    return ''.join(out), warnings


# Cross-reference linking: turn "Phase N" / "phaseN" prose mentions into
# clickable anchors pointing to the corresponding nav target. We skip code
# blocks, inline code, and the SVG mermaid wrappers we just produced.
_XREF_PHASE_RE = re.compile(r'(?<!\[)\bPhase\s*([0-8])\b(?![\]0-9])')
_XREF_BASICS_RE = re.compile(r'(?<!\[)(序章|训练基础)(?![\]])')


def add_xref_links(md: str) -> str:
    parts = re.split(
        r'(```[\s\S]*?```|`[^`\n]+`|<div class="mermaid-block">[\s\S]*?</div>)',
        md,
    )
    out = []
    for part in parts:
        if (part.startswith('```')
                or (part.startswith('`') and not part.startswith('```'))
                or part.startswith('<div class="mermaid-block"')):
            out.append(part)
            continue
        # Process line by line so we skip headings & table separators
        lines = part.split('\n')
        for i, line in enumerate(lines):
            stripped = line.lstrip()
            if not stripped or stripped.startswith('#') or stripped.startswith('|---'):
                continue
            # Don't link inside markdown link text or url
            line = _XREF_PHASE_RE.sub(r'[Phase \1](#phase\1)', line)
            lines[i] = line
        out.append('\n'.join(lines))
    return ''.join(out)


# When README is rendered as the site's "总览" chapter, the "▶ 在线阅读" badge
# is reversed (the reader is already on the site). Swap it for a CTA that links
# back to the GitHub source.
_README_SITE_CTA_OLD = (
    '[![Live Site](https://img.shields.io/badge/'
    '%E2%96%B6_%E5%9C%A8%E7%BA%BF%E9%98%85%E8%AF%BB-a52619?'
    'style=for-the-badge)](https://sqhuang.github.io/coding-llm-handbook/)'
)
_README_SITE_CTA_NEW = (
    '[![GitHub Repo](https://img.shields.io/badge/'
    '%E2%96%B6_GitHub_%E4%BB%93%E5%BA%93-1c1814?'
    'style=for-the-badge&logo=github&logoColor=white)]'
    '(https://github.com/sqhuang/coding-llm-handbook)'
)


def adapt_readme_for_site(md: str) -> str:
    """Replace the 'Live Site' CTA with an icon-only GitHub link when on the site.

    The README serves both as the GitHub repo landing page AND as the site's
    总览 chapter. The two contexts need flipped CTAs. On the site we render a
    minimal icon-only octocat that links back to the repo source.
    """
    patterns = [
        '[![Live Site](https://img.shields.io/badge/▶_在线阅读-a52619?style=for-the-badge)](https://sqhuang.github.io/coding-llm-handbook/)',
    ]
    # Icon-only octocat from SimpleIcons CDN — two color variants so the
    # logo stays visible in both light and dark site themes.
    replacement = (
        '<a class="gh-icon-link" href="https://github.com/sqhuang/coding-llm-handbook" '
        'aria-label="GitHub" title="View source on GitHub">'
        '<img class="gh-icon gh-icon-light" src="https://cdn.simpleicons.org/github/1c1814" '
        'height="36" alt="GitHub"/>'
        '<img class="gh-icon gh-icon-dark" src="https://cdn.simpleicons.org/github/e8dfcb" '
        'height="36" alt="GitHub"/>'
        '</a>'
    )
    for p in patterns:
        if p in md:
            md = md.replace(p, replacement)
            break
    return md


# ---------- Anchor / link checker

# Internal anchor: `[label](#xxxx)` not crossing a file boundary.
_INTERNAL_ANCHOR_RE = re.compile(r'\]\(#([A-Za-z][A-Za-z0-9_\-]*)\)')
# Remaining `](./...md)` after rewrite = broken local link.
_RESIDUAL_LOCAL_MD_RE = re.compile(r'\]\(\./[^)]*\.md[^)]*\)')


def check_links(content: dict[str, str]) -> tuple[int, list[str]]:
    """Validate that every internal anchor + cross-md link resolves.

    Returns (external_link_count, errors). Caller decides whether to abort.
    """
    errors: list[str] = []
    external_n = 0

    # Authorable anchors are nav ids (top-level chapter targets). Heading-level
    # h-N ids are runtime-generated by app.js, so we don't accept them as link
    # targets in source markdown.
    valid = set(NAV_IDS)

    for cid, md in content.items():
        # Skip code blocks
        parts = re.split(r'(```[\s\S]*?```|`[^`\n]+`)', md)
        prose = ''.join(
            p for p in parts
            if not (p.startswith('```') or (p.startswith('`') and not p.startswith('```')))
        )
        for m in _INTERNAL_ANCHOR_RE.finditer(prose):
            tgt = m.group(1)
            if tgt not in valid:
                errors.append(f"[{cid}] unknown anchor #{tgt}")
        for m in _RESIDUAL_LOCAL_MD_RE.finditer(prose):
            errors.append(f"[{cid}] residual local-md link: {m.group(0)}")
        external_n += len(re.findall(r'\]\(https?://', prose))
    return external_n, errors


def estimate_reading_time(md: str) -> int:
    """Rough reading time in minutes: 350 CJK / min + 220 words / min."""
    cjk = len(re.findall(r'[一-鿿]', md))
    words = len(re.findall(r'\b\w+\b', re.sub(r'[一-鿿]', ' ', md)))
    return max(1, round(cjk / 350 + words / 220))


def build():
    print("→ Loading template assets")
    template = (ASSETS / "template.html").read_text(encoding="utf-8")
    style_css = (ASSETS / "style.css").read_text(encoding="utf-8")
    app_js = (ASSETS / "app.js").read_text(encoding="utf-8")

    print("→ Reading markdown + pre-rendering Mermaid blocks")
    if not MMDC.exists():
        print(f"  ! mmdc not found at {MMDC} — run: npm install --save-dev @mermaid-js/mermaid-cli")
        return

    # Pass 1 — load raw, run text-level transforms (includes + cross-md link
    # rewrite). These run before mermaid so that included files can themselves
    # contain mermaid blocks, and so that link rewrites apply to included text.
    raw: dict[str, str] = {}
    total_includes = 0
    include_errors: list[str] = []
    link_warnings: list[str] = []
    for k, v in FILES.items():
        md = (HERE / v).read_text(encoding="utf-8")
        if k == "readme":
            md = adapt_readme_for_site(md)
        md, n_inc, errs = process_includes(md)
        if n_inc:
            print(f"  · {k}: {n_inc} include directive(s)")
        total_includes += n_inc
        include_errors.extend(errs)
        md, warns = rewrite_local_md_links(md)
        link_warnings.extend(warns)
        raw[k] = md

    # Pass 2 — pre-warm mermaid cache for any uncached blocks (parallel)
    prewarm_mermaid_cache(list(raw.values()))

    # Pass 3 — per-chapter mermaid substitution + xrefs + reading time
    content: dict[str, str] = {}
    reading_times: dict[str, int] = {}
    total_mermaid = 0
    total_failed = 0
    for k, md in raw.items():
        n_mermaid = len(MERMAID_RE.findall(md))
        if n_mermaid:
            print(f"  · {k}: {n_mermaid} mermaid block(s)")
            md, n, fail = replace_mermaid_blocks(md)
            total_mermaid += n
            total_failed += fail
        # Reading time should be based on prose length, before xref bloat
        reading_times[k] = estimate_reading_time(md)
        # Convert "Phase N" / cross-refs into clickable anchors
        md = add_xref_links(md)
        content[k] = md

    # Pass 4 — link validation across the assembled corpus
    external_n, link_errors = check_links(content)
    if include_errors or link_errors:
        print("✗ link/include check FAILED:")
        for e in include_errors + link_errors:
            print(f"    {e}")
        sys.exit(1)
    if link_warnings:
        print("⚠ link warnings:")
        for w in link_warnings:
            print(f"    {w}")
    print(f"  · links ok ({external_n} external)")

    print("→ Assembling HTML")
    nav_json = json.dumps(
        [{"id": n[0], "label": n[1], "num": n[2], "title": n[3]} for n in NAV],
        ensure_ascii=False,
    )
    # Escape `</` to keep JSON intact inside <script>...</script>
    content_json = json.dumps(content, ensure_ascii=False).replace("</", "<\\/")
    meta_json = json.dumps({
        "readingTime": reading_times,
        "buildTime":   datetime.datetime.now().isoformat(timespec="seconds"),
    }, ensure_ascii=False)

    html = template
    html = html.replace("/*__STYLE_CSS__*/",   style_css)
    html = html.replace("/*__APP_JS__*/",      app_js)
    html = html.replace("/*__NAV_JSON__*/",    nav_json)
    html = html.replace("/*__CONTENT_JSON__*/", content_json)
    html = html.replace("/*__META_JSON__*/",   meta_json)

    out = HERE / "index.html"
    out.write_text(html, encoding="utf-8")

    summary = f"wrote {out.relative_to(HERE)} ({out.stat().st_size // 1024} KB)"
    summary += f" · {total_mermaid} mermaid · {total_includes} include · {external_n} ext-link"
    if total_failed:
        summary += f" · {total_failed} FAILED"
    print(f"✓ {summary}")


if __name__ == "__main__":
    build()
