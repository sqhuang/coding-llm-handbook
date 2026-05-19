(() => {
  const NAV     = JSON.parse(document.getElementById("nav-data").textContent);
  const CONTENT = JSON.parse(document.getElementById("content-data").textContent);
  const META    = JSON.parse(document.getElementById("meta-data").textContent || "{}");

  // ---- short English label for sidebar ----
  const SHORT_EN = {
    readme:     "Overview",
    roadmap:    "Roadmap",
    basics:     "Basics",
    phase0:     "Foundation",
    phase1:     "Data",
    phase2:     "Architecture",
    phase3:     "Long-Ctx",
    phase4:     "SFT",
    phase5:     "RL",
    phase6:     "Eval",
    phase7:     "Deploy",
    phase8:     "Agent",
    lab:        "Lab",
    capstone:   "Capstone",
    outro:      "What's Next",
    glossary:   "Glossary",
    references: "References",
  };

  // ---- localStorage keys + read state ----
  const RS_KEY = "readstate:v1";
  const SCROLL_KEY = (id) => `scroll:${id}`;
  const READ_THRESHOLD = 0.8; // 80% scrolled => mark read

  function loadReadState() {
    try { return JSON.parse(localStorage.getItem(RS_KEY) || "{}"); }
    catch { return {}; }
  }
  function saveReadState(s) {
    try { localStorage.setItem(RS_KEY, JSON.stringify(s)); } catch {}
  }
  let READ = loadReadState();
  function markChapter(id, state) {
    if (READ[id] === state) return;
    READ[id] = state;
    saveReadState(READ);
    refreshNavState();
    refreshOverallProgress();
  }
  // glyphs
  const STATE_GLYPH = { read: "✓", reading: "◐", unread: "·" };

  // ---- marked setup ----
  const renderer = new marked.Renderer();
  marked.setOptions({
    renderer,
    breaks: false,
    gfm: true,
    highlight: (code, lang) => {
      try {
        if (lang && hljs.getLanguage(lang)) {
          return hljs.highlight(code, { language: lang }).value;
        }
        return hljs.highlightAuto(code).value;
      } catch {
        return code;
      }
    },
  });

  // ---- Math (KaTeX) preprocess/postprocess ----
  function preprocessMath(md) {
    const math = [];
    const parts = md.split(/(```[\s\S]*?```|`[^`\n]+`)/g);
    const processed = parts.map(part => {
      if (part.startsWith('```') || (part.startsWith('`') && !part.startsWith('```'))) {
        return part;
      }
      part = part.replace(/\$\$([\s\S]+?)\$\$/g, (_, expr) => {
        math.push({ display: true, expr: expr.trim() });
        return `@@KTX${math.length - 1}@@`;
      });
      part = part.replace(/(?<!\\)\$([^$\n]+?)(?<!\\)\$/g, (_, expr) => {
        math.push({ display: false, expr: expr.trim() });
        return `@@KTX${math.length - 1}@@`;
      });
      return part;
    });
    return { md: processed.join(''), math };
  }
  function postprocessMath(html, math) {
    if (!math.length) return html;
    return html.replace(/@@KTX(\d+)@@/g, (_, i) => {
      const m = math[+i];
      if (!window.katex) return `<code>${escapeHtml(m.expr)}</code>`;
      try {
        return katex.renderToString(m.expr, {
          displayMode: m.display,
          throwOnError: false,
          strict: "ignore",
          output: "htmlAndMathml",
        });
      } catch (e) {
        return `<code>${escapeHtml(m.expr)}</code>`;
      }
    });
  }
  function escapeHtml(s) {
    return s.replace(/[&<>"']/g, c => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
    }[c]));
  }
  function escapeReg(s) { return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); }

  // ---- Reading time ----
  function readingTime(text) {
    const cjk = (text.match(/[一-鿿㐀-䶿]/g) || []).length;
    const words = (text.replace(/[一-鿿㐀-䶿]/g, " ").match(/\b\w+\b/g) || []).length;
    const minutes = Math.max(1, Math.round(cjk / 350 + words / 220));
    return minutes;
  }
  // Approximate reading time of arbitrary text snippet (for per-section TOC)
  function readingTimeOf(text) {
    const cjk = (text.match(/[一-鿿㐀-䶿]/g) || []).length;
    const words = (text.replace(/[一-鿿㐀-䶿]/g, " ").match(/\b\w+\b/g) || []).length;
    const minutes = cjk / 350 + words / 220;
    return minutes < 0.95 ? "<1m" : Math.round(minutes) + "m";
  }

  // ---- Nav (sidebar) ----
  function navStateGlyph(id) {
    return STATE_GLYPH[READ[id] || "unread"];
  }
  function renderNav() {
    const list = document.getElementById("nav-list");
    list.innerHTML = NAV.map((item, i) => {
      let sep = "";
      if (item.id === "phase0") {
        sep = `<li class="nav-section" role="separator"><span class="glyph">·</span> 正文 · Chapters</li>`;
      } else if (item.id === "lab") {
        sep = `<li class="nav-section" role="separator"><span class="glyph">·</span> 附录 · Appendix</li>`;
      }
      const delay = (i * 0.025).toFixed(2);
      const state = READ[item.id] || "unread";
      return `${sep}<li>
        <a class="nav-link" data-target="${item.id}" data-read="${state}" href="#${item.id}" style="animation-delay:${delay}s">
          <span class="state" aria-hidden="true">${navStateGlyph(item.id)}</span>
          <span class="num">${item.num}</span>
          <span class="label">${item.label}</span>
          <span class="title-en">${SHORT_EN[item.id] || ""}</span>
        </a>
      </li>`;
    }).join("");
    list.querySelectorAll(".nav-link").forEach(a => {
      a.addEventListener("click", e => {
        e.preventDefault();
        load(a.dataset.target);
      });
    });
  }
  function refreshNavState() {
    document.querySelectorAll(".nav-link").forEach(a => {
      const id = a.dataset.target;
      a.dataset.read = READ[id] || "unread";
      const stateEl = a.querySelector(".state");
      if (stateEl) stateEl.textContent = navStateGlyph(id);
    });
  }

  // ---- Page nav (prev / next) ----
  function renderPageNav(currentId) {
    const idx = NAV.findIndex(n => n.id === currentId);
    const prev = idx > 0 ? NAV[idx - 1] : null;
    const next = idx < NAV.length - 1 ? NAV[idx + 1] : null;
    const el = document.getElementById("page-nav");
    el.innerHTML = `
      ${prev
        ? `<a class="prev" data-target="${prev.id}" href="#${prev.id}">
             <span class="nav-hint">← 上一章 · ${prev.num}</span>
             <span class="nav-title">${prev.label} · ${truncate(prev.title, 30)}</span>
           </a>`
        : `<span class="prev placeholder"><span class="nav-hint">序章</span><span class="nav-title">已是第一篇</span></span>`}
      ${next
        ? `<a class="next" data-target="${next.id}" href="#${next.id}">
             <span class="nav-hint">下一章 · ${next.num} →</span>
             <span class="nav-title">${next.label} · ${truncate(next.title, 30)}</span>
           </a>`
        : `<span class="next placeholder"><span class="nav-hint">完结</span><span class="nav-title">已是最后一篇</span></span>`}
    `;
    el.querySelectorAll("a[data-target]").forEach(a => {
      a.addEventListener("click", e => {
        e.preventDefault();
        load(a.dataset.target);
      });
    });
  }
  function truncate(s, n) { return s.length > n ? s.slice(0, n - 1) + "…" : s; }

  // ---- Scroll position persistence ----
  let activeId = null;
  let scrollSaveTimer = null;
  function saveScroll() {
    if (!activeId) return;
    clearTimeout(scrollSaveTimer);
    scrollSaveTimer = setTimeout(() => {
      try { localStorage.setItem(SCROLL_KEY(activeId), String(window.scrollY)); }
      catch {}
    }, 200);
  }
  function restoreScroll(id) {
    const saved = parseInt(localStorage.getItem(SCROLL_KEY(id)) || "0", 10);
    if (saved > 0) {
      requestAnimationFrame(() => window.scrollTo({ top: saved, behavior: "instant" }));
      return true;
    }
    return false;
  }

  // ---- Per-chapter rendered HTML cache ----
  const RENDERED = {};
  function renderChapter(id) {
    if (RENDERED[id]) return RENDERED[id];
    if (!CONTENT[id]) return "";
    const { md, math } = preprocessMath(CONTENT[id]);
    let html = marked.parse(md);
    html = postprocessMath(html, math);
    RENDERED[id] = html;
    return html;
  }
  function prefetchAdjacent(id) {
    const idle = window.requestIdleCallback || (cb => setTimeout(cb, 200));
    const idx = NAV.findIndex(n => n.id === id);
    const neighbors = [NAV[idx - 1], NAV[idx + 1]].filter(Boolean);
    neighbors.forEach(n => {
      if (!RENDERED[n.id]) idle(() => renderChapter(n.id), { timeout: 2000 });
    });
  }

  // ---- Callout detection (blockquotes that start with ⚡/📌/⚠️/📅) ----
  const CALLOUT_MAP = [
    { re: /^[⚡]/u,                    type: "tip",   label: "要点",     icon: "⚡" },
    { re: /^📌/u,                      type: "check", label: "章末检查", icon: "📌" },
    { re: /^[⚠️]|^⚠/u,                 type: "warn",  label: "常见坑",   icon: "⚠️" },
    { re: /^📅/u,                      type: "date",  label: "主线快照", icon: "📅" },
  ];
  function decorateCallouts(root) {
    root.querySelectorAll("blockquote").forEach(bq => {
      // Use the first non-whitespace text of the blockquote to detect emoji
      const text = (bq.textContent || "").trim();
      for (const c of CALLOUT_MAP) {
        if (c.re.test(text)) {
          bq.setAttribute("data-callout", c.type);
          bq.setAttribute("data-icon", c.icon);
          bq.setAttribute("data-label", c.label);
          // Strip the leading emoji from first <strong>/<p> so it isn't duplicated
          const first = bq.querySelector("p, li, strong");
          if (first) {
            first.innerHTML = first.innerHTML.replace(c.re, "").replace(/^\s+/, "");
          }
          break;
        }
      }
    });
  }

  // ---- Code-block enhancements: language tag + line numbers ----
  function decorateCode(root) {
    root.querySelectorAll("pre").forEach(pre => {
      const code = pre.querySelector("code");
      if (!code) return;
      // Detect language from hljs class or marked's `language-xxx`
      let lang = "";
      (code.className || "").split(/\s+/).forEach(cls => {
        if (cls.startsWith("language-")) lang = cls.slice("language-".length);
        else if (cls === "hljs") {}
        else if (cls && !lang) lang = cls;
      });
      lang = (lang || "text").replace(/^hljs-?/, "");
      if (lang.length > 12) lang = "text";

      const tag = document.createElement("span");
      tag.className = "code-tag";
      tag.textContent = lang;
      pre.appendChild(tag);

      // Line numbers: only for languages where lines make sense (skip plain text/diff sometimes)
      if (lang !== "text" || code.textContent.split("\n").length > 4) {
        wrapLines(code);
        pre.classList.add("has-lines");
      }

      // Copy button (always visible at 0.55 opacity; full on hover)
      const btn = document.createElement("button");
      btn.className = "copy-btn";
      btn.textContent = "复制";
      btn.addEventListener("click", () => {
        navigator.clipboard.writeText(code.textContent);
        btn.textContent = "已复制";
        btn.classList.add("done");
        setTimeout(() => { btn.textContent = "复制"; btn.classList.remove("done"); }, 1600);
      });
      pre.appendChild(btn);
    });
  }
  function wrapLines(code) {
    // Walk children, splitting on \n, wrapping each line in <span class="line">.
    // We need to preserve highlight.js's nested spans, so we re-tokenize at the
    // text-node level: traverse, find newlines inside text nodes, split into
    // line spans.
    const lines = [];
    let current = document.createElement("span");
    current.className = "line";
    function flushCurrent() {
      lines.push(current);
      current = document.createElement("span");
      current.className = "line";
    }
    function visit(node, parent) {
      if (node.nodeType === Node.TEXT_NODE) {
        const parts = node.nodeValue.split("\n");
        parts.forEach((part, i) => {
          if (part) {
            // If parent is a styled span, clone it (without children) and add text.
            if (parent && parent !== code) {
              const clone = parent.cloneNode(false);
              clone.appendChild(document.createTextNode(part));
              current.appendChild(clone);
            } else {
              current.appendChild(document.createTextNode(part));
            }
          }
          if (i < parts.length - 1) flushCurrent();
        });
        return;
      }
      if (node.nodeType !== Node.ELEMENT_NODE) return;
      // Recurse into children, treating this element as the "parent" for inheritance
      Array.from(node.childNodes).forEach(child => visit(child, node));
    }
    Array.from(code.childNodes).forEach(child => visit(child, null));
    flushCurrent();
    code.innerHTML = "";
    lines.forEach(l => code.appendChild(l));
  }

  // ---- Load phase ----
  function load(id, push = true) {
    const item = NAV.find(n => n.id === id);
    if (!item || !CONTENT[id]) return;

    activeId = id;
    // Mark as 'reading' as soon as you open it (unless already 'read')
    if (READ[id] !== "read") markChapter(id, "reading");

    const content = document.getElementById("content");
    content.style.animation = "none";
    void content.offsetHeight;
    content.style.animation = "";

    content.innerHTML = renderChapter(id);

    decorateCallouts(content);
    decorateCode(content);

    const minutes = (META.readingTime && META.readingTime[id]) || readingTime(CONTENT[id]);
    document.getElementById("breadcrumb").innerHTML =
      `<span>${item.num} · ${item.title}</span><span class="reading-time">约 ${minutes} 分钟</span>`;

    document.querySelectorAll(".nav-link").forEach(a => {
      a.classList.toggle("active", a.dataset.target === id);
    });
    document.title = `${item.title} · 研究手札`;

    content.querySelectorAll("pre code").forEach(b => {
      try { hljs.highlightElement(b); } catch {}
    });

    // Heading anchors + ids
    let hIdx = 0;
    const sectionInfos = [];
    content.querySelectorAll("h2, h3").forEach(h => {
      const hid = "h-" + (hIdx++);
      h.id = hid;
      const a = document.createElement("a");
      a.href = "#" + hid;
      a.className = "anchor";
      a.textContent = "§";
      a.addEventListener("click", e => {
        e.preventDefault();
        document.getElementById(hid).scrollIntoView({ behavior: "smooth", block: "start" });
      });
      h.appendChild(a);
      sectionInfos.push({ id: hid, el: h });
    });

    buildToc(sectionInfos);
    renderPageNav(id);
    setStatusbarPosition(item, null);
    if (push) history.replaceState(null, "", "#" + id);

    if (!restoreScroll(id)) {
      window.scrollTo({ top: 0, behavior: "instant" });
    }

    prefetchAdjacent(id);
    refreshOverallProgress();
  }

  // ---- TOC mini-map (with per-section reading time + current dot) ----
  function buildToc(sectionInfos) {
    const toc = document.getElementById("toc");
    const hs = sectionInfos.map(s => s.el);
    if (hs.length < 3) {
      toc.classList.remove("show");
      toc.innerHTML = "";
      return;
    }
    // Compute reading time per H2 (text between consecutive H2s)
    const times = sectionTimes(hs);

    let html = `<div class="toc-title"><span>本页目录</span><span class="toc-count">${hs.length}</span></div><ul class="toc-list">`;
    hs.forEach((h, i) => {
      const cls = h.tagName === "H3" ? "lvl-3" : "";
      const text = h.textContent.replace(/§$/, "").trim();
      const time = times[i] || "";
      html += `<li><a class="${cls}" href="#${h.id}"><span class="toc-text">${escapeHtml(text)}</span>${time ? `<span class="toc-time">${time}</span>` : ""}</a></li>`;
    });
    html += `</ul>`;
    toc.innerHTML = html;
    toc.classList.add("show");

    toc.querySelectorAll("a").forEach(a => {
      a.addEventListener("click", e => {
        e.preventDefault();
        const target = document.getElementById(a.getAttribute("href").slice(1));
        if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
      });
    });

    if (window.__tocIO) window.__tocIO.disconnect();
    const io = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const id = entry.target.id;
          toc.querySelectorAll("a").forEach(a => {
            a.classList.toggle("active", a.getAttribute("href") === "#" + id);
          });
          // Update status bar section text
          const sb = document.getElementById("sb-section");
          const text = entry.target.textContent.replace(/§$/, "").trim();
          sb.textContent = text;
          sb.classList.toggle("empty", !text);
        }
      });
    }, { rootMargin: "-10% 0px -75% 0px" });
    hs.forEach(h => io.observe(h));
    window.__tocIO = io;
  }
  function sectionTimes(hs) {
    // Estimate reading time of text between hs[i] and hs[i+1] (or end of content)
    const content = document.getElementById("content");
    const times = [];
    for (let i = 0; i < hs.length; i++) {
      if (hs[i].tagName === "H3") { times.push(""); continue; }
      let buf = "";
      let node = hs[i].nextSibling;
      const stopAt = hs[i + 1] || null;
      while (node && node !== stopAt) {
        if (node.nodeType === Node.TEXT_NODE) buf += node.nodeValue;
        else if (node.nodeType === Node.ELEMENT_NODE) buf += node.textContent || "";
        node = node.nextSibling;
      }
      times.push(buf.length > 100 ? readingTimeOf(buf) : "");
    }
    return times;
  }

  // ---- Statusbar ----
  function setStatusbarPosition(item, sectionText) {
    document.getElementById("sb-pos").textContent = `${item.num} · ${item.label}`;
    const sec = document.getElementById("sb-section");
    sec.textContent = sectionText || "";
    sec.classList.toggle("empty", !sectionText);
  }
  function setStatusbarMode() {
    const accent = document.documentElement.dataset.accent || "purple";
    document.getElementById("sb-mode").textContent = "midnight · " + accent;
  }
  function refreshOverallProgress() {
    const total = NAV.length;
    const read = NAV.filter(n => READ[n.id] === "read").length;
    const reading = NAV.filter(n => READ[n.id] === "reading").length;
    const pct = Math.round((read + reading * 0.4) / total * 100);
    document.getElementById("sb-overall-fill").style.width = pct + "%";
    document.getElementById("sb-overall-label").textContent =
      `${read}/${total} 章 · ${pct}%`;
    // Total minutes left = sum reading time of unread + half of reading
    const tt = META.readingTime || {};
    let mins = 0;
    NAV.forEach(n => {
      const t = tt[n.id] || 0;
      const s = READ[n.id];
      if (s === "read") return;
      mins += s === "reading" ? t * 0.5 : t;
    });
    document.getElementById("sb-overall-time").textContent =
      mins > 0 ? `≈ ${Math.round(mins)} 分钟剩余` : "✓ 全部读完";
  }

  // ---- Accent (replaces old light/dark theme; we are dark-only now) ----
  const ACCENTS = ["purple", "cyan", "amber"];
  document.getElementById("theme-btn").addEventListener("click", () => {
    const html = document.documentElement;
    const curr = html.dataset.accent || "purple";
    const next = ACCENTS[(ACCENTS.indexOf(curr) + 1) % ACCENTS.length];
    html.dataset.accent = next;
    localStorage.setItem("accent", next);
    setStatusbarMode();
  });

  // ---- Font size ----
  const SIZES = ["small", "medium", "large"];
  document.getElementById("font-btn").addEventListener("click", () => {
    const curr = document.documentElement.dataset.fontsize || "medium";
    const next = SIZES[(SIZES.indexOf(curr) + 1) % 3];
    document.documentElement.dataset.fontsize = next;
    localStorage.setItem("fontsize", next);
  });

  // ---- Density ----
  document.getElementById("density-btn").addEventListener("click", () => {
    const curr = document.documentElement.dataset.density || "compact";
    const next = curr === "compact" ? "comfy" : "compact";
    document.documentElement.dataset.density = next;
    localStorage.setItem("density", next);
  });

  // ---- Print ----
  document.getElementById("print-btn").addEventListener("click", () => {
    window.print();
  });

  // ---- Keyboard shortcuts ----
  document.addEventListener("keydown", e => {
    const inForm = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement.tagName);
    if (inForm) {
      if (e.key === "Escape") document.activeElement.blur();
      return;
    }
    if (e.key === "/")     { e.preventDefault(); document.getElementById("search-input").focus(); }
    if (e.key === "t")     { document.getElementById("theme-btn").click(); }
    if (e.key === "a")     { document.getElementById("font-btn").click(); }
    if (e.key === "d")     { document.getElementById("density-btn").click(); }
    if (e.key === "j" || e.key === "ArrowRight") { goNext(); }
    if (e.key === "k" || e.key === "ArrowLeft")  { goPrev(); }
  });
  function goPrev() {
    const idx = NAV.findIndex(n => n.id === activeId);
    if (idx > 0) load(NAV[idx - 1].id);
  }
  function goNext() {
    const idx = NAV.findIndex(n => n.id === activeId);
    if (idx >= 0 && idx < NAV.length - 1) load(NAV[idx + 1].id);
  }

  // ---- Search ----
  const input = document.getElementById("search-input");
  const results = document.getElementById("search-results");
  let searchT = null;
  function doSearch(q) {
    q = q.trim();
    if (q.length < 2) { results.classList.remove("show"); return; }
    const needle = q.toLowerCase();
    const hits = [];
    for (const [id, text] of Object.entries(CONTENT)) {
      const lower = text.toLowerCase();
      let from = 0, count = 0;
      while (count < 2) {
        const idx = lower.indexOf(needle, from);
        if (idx < 0) break;
        const s = Math.max(0, idx - 40);
        const e = Math.min(text.length, idx + q.length + 90);
        let snip = text.substring(s, e).replace(/[\n\r]+/g, " ").replace(/\s+/g, " ");
        snip = escapeHtml(snip).replace(new RegExp(escapeReg(q), "gi"), m => `<mark>${m}</mark>`);
        const item = NAV.find(n => n.id === id);
        hits.push({
          id,
          title: item ? `${item.num} · ${item.title}` : id,
          snippet: (s > 0 ? "…" : "") + snip + (e < text.length ? "…" : ""),
        });
        from = idx + q.length;
        count++;
      }
    }
    if (!hits.length) {
      results.innerHTML = `<div class="search-result"><div class="search-result-snippet" style="color:var(--text-muted)">无匹配</div></div>`;
    } else {
      results.innerHTML = hits.map(h => `
        <div class="search-result" data-id="${h.id}">
          <div class="search-result-title">${h.title}</div>
          <div class="search-result-snippet">${h.snippet}</div>
        </div>`).join("");
      results.querySelectorAll(".search-result[data-id]").forEach(el => {
        el.addEventListener("click", () => {
          load(el.dataset.id);
          results.classList.remove("show");
          input.value = "";
        });
      });
    }
    results.classList.add("show");
  }
  input.addEventListener("input", e => {
    clearTimeout(searchT);
    searchT = setTimeout(() => doSearch(e.target.value), 120);
  });
  input.addEventListener("focus", e => {
    if (e.target.value.length >= 2) doSearch(e.target.value);
  });
  document.addEventListener("click", e => {
    if (!e.target.closest(".search")) results.classList.remove("show");
  });

  // ---- Scroll: per-chapter progress + statusbar bar + read-state advance ----
  const sbBar = document.getElementById("sb-bar-fill");
  const sbPct = document.getElementById("sb-pct");
  function updateScroll() {
    const h = document.documentElement;
    const max = h.scrollHeight - h.clientHeight;
    const pct = max > 0 ? (h.scrollTop / max) : 0;
    const pctClamped = Math.min(1, Math.max(0, pct));
    const pctNum = Math.round(pctClamped * 100);
    sbBar.style.width = pctNum + "%";
    sbPct.textContent = pctNum + "%";
    if (activeId && pctClamped >= READ_THRESHOLD && READ[activeId] !== "read") {
      markChapter(activeId, "read");
    }
  }
  window.addEventListener("scroll", () => {
    updateScroll();
    saveScroll();
  }, { passive: true });

  // ---- Restore prefs ----
  // Dark theme is forced; we no longer respect a saved light/dark.
  document.documentElement.dataset.theme = "dark";
  document.getElementById("hljs-light").disabled = true;
  document.getElementById("hljs-dark").disabled  = false;
  const savedAccent = localStorage.getItem("accent");
  if (savedAccent && ACCENTS.includes(savedAccent)) {
    document.documentElement.dataset.accent = savedAccent;
  }
  const savedFont = localStorage.getItem("fontsize");
  if (savedFont) document.documentElement.dataset.fontsize = savedFont;
  const savedDensity = localStorage.getItem("density");
  if (savedDensity) document.documentElement.dataset.density = savedDensity;

  // ---- Init ----
  renderNav();
  setStatusbarMode();
  refreshOverallProgress();
  const initial = (location.hash || "#readme").replace(/^#/, "");
  load(NAV.find(n => n.id === initial) ? initial : "readme", false);
  window.addEventListener("hashchange", () => {
    const id = location.hash.replace(/^#/, "");
    if (NAV.find(n => n.id === id) && id !== activeId) load(id, false);
  });
})();
