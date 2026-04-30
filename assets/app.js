(() => {
  const NAV     = JSON.parse(document.getElementById("nav-data").textContent);
  const CONTENT = JSON.parse(document.getElementById("content-data").textContent);
  const META    = JSON.parse(document.getElementById("meta-data").textContent || "{}");

  // Map id -> short English label for sidebar third column
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
    outro:      "What's Next",
    glossary:   "Glossary",
    references: "References",
  };

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
    // Count CJK chars + non-CJK words
    const cjk = (text.match(/[一-鿿㐀-䶿]/g) || []).length;
    const words = (text.replace(/[一-鿿㐀-䶿]/g, " ").match(/\b\w+\b/g) || []).length;
    // ~350 cjk/min, ~220 words/min
    const minutes = Math.max(1, Math.round(cjk / 350 + words / 220));
    return minutes;
  }

  // ---- Render nav ----
  function renderNav() {
    const list = document.getElementById("nav-list");
    list.innerHTML = NAV.map((item, i) => {
      let sep = "";
      if (item.id === "phase0") {
        sep = `<li class="nav-section" role="separator"><span class="glyph">·</span> 正文 · Chapters</li>`;
      } else if (item.id === "lab") {
        sep = `<li class="nav-section" role="separator"><span class="glyph">·</span> 附录 · Appendix</li>`;
      }
      const delay = (i * 0.03).toFixed(2);
      return `${sep}<li>
        <a class="nav-link" data-target="${item.id}" href="#${item.id}" style="animation-delay:${delay}s">
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
  const SCROLL_KEY = (id) => `scroll:${id}`;
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
      // wait one frame for layout
      requestAnimationFrame(() => window.scrollTo({ top: saved, behavior: "instant" }));
      return true;
    }
    return false;
  }

  // ---- Load phase ----
  function load(id, push = true) {
    const item = NAV.find(n => n.id === id);
    if (!item || !CONTENT[id]) return;

    activeId = id;
    const content = document.getElementById("content");
    content.style.animation = "none";
    void content.offsetHeight;
    content.style.animation = "";

    const { md, math } = preprocessMath(CONTENT[id]);
    let html = marked.parse(md);
    html = postprocessMath(html, math);
    content.innerHTML = html;

    // breadcrumb (with reading time)
    const minutes = (META.readingTime && META.readingTime[id]) || readingTime(CONTENT[id]);
    document.getElementById("breadcrumb").innerHTML =
      `<span>${item.num} · ${item.title}</span><span class="reading-time">约 ${minutes} 分钟</span>`;

    document.querySelectorAll(".nav-link").forEach(a => {
      a.classList.toggle("active", a.dataset.target === id);
    });
    document.title = `${item.title} · 研究手札`;

    // hljs (mermaid blocks are already pre-rendered SVG inline, no client-side mermaid)
    content.querySelectorAll("pre code").forEach(b => {
      try { hljs.highlightElement(b); } catch {}
    });

    // copy buttons
    content.querySelectorAll("pre").forEach(pre => {
      const btn = document.createElement("button");
      btn.className = "copy-btn";
      btn.textContent = "复制";
      btn.addEventListener("click", () => {
        const code = pre.querySelector("code");
        if (code) {
          navigator.clipboard.writeText(code.textContent);
          btn.textContent = "已复制";
          btn.classList.add("done");
          setTimeout(() => { btn.textContent = "复制"; btn.classList.remove("done"); }, 1600);
        }
      });
      pre.appendChild(btn);
    });

    // heading anchors
    let hIdx = 0;
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
    });

    buildToc();
    renderPageNav(id);
    if (push) history.replaceState(null, "", "#" + id);

    // restore scroll position if any, else top
    if (!restoreScroll(id)) {
      window.scrollTo({ top: 0, behavior: "instant" });
    }
  }

  // ---- TOC ----
  function buildToc() {
    const toc = document.getElementById("toc");
    const hs = document.querySelectorAll("#content h2, #content h3");
    if (hs.length < 3) {
      toc.classList.remove("show");
      toc.innerHTML = "";
      return;
    }
    let html = `<div class="toc-title">本页目录</div><ul class="toc-list">`;
    hs.forEach(h => {
      const cls = h.tagName === "H3" ? "lvl-3" : "";
      const text = h.textContent.replace(/§$/, "").trim();
      html += `<li><a class="${cls}" href="#${h.id}">${text}</a></li>`;
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

    const io = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const id = entry.target.id;
          toc.querySelectorAll("a").forEach(a => {
            a.classList.toggle("active", a.getAttribute("href") === "#" + id);
          });
        }
      });
    }, { rootMargin: "-10% 0px -75% 0px" });
    hs.forEach(h => io.observe(h));
  }

  // ---- Theme ----
  document.getElementById("theme-btn").addEventListener("click", () => {
    const html = document.documentElement;
    const next = html.dataset.theme === "dark" ? "light" : "dark";
    html.dataset.theme = next;
    document.getElementById("hljs-light").disabled = (next === "dark");
    document.getElementById("hljs-dark").disabled  = (next === "light");
    localStorage.setItem("theme", next);
  });

  // ---- Font size ----
  const SIZES = ["small", "medium", "large"];
  document.getElementById("font-btn").addEventListener("click", () => {
    const curr = document.documentElement.dataset.fontsize || "medium";
    const next = SIZES[(SIZES.indexOf(curr) + 1) % 3];
    document.documentElement.dataset.fontsize = next;
    localStorage.setItem("fontsize", next);
  });

  // ---- Print ----
  document.getElementById("print-btn").addEventListener("click", () => {
    window.print();
  });

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
  document.addEventListener("keydown", e => {
    if (e.key === "/" && document.activeElement !== input) {
      e.preventDefault();
      input.focus();
      input.select();
    }
    if (e.key === "Escape") {
      if (document.activeElement === input) input.blur();
      results.classList.remove("show");
    }
  });

  // ---- Progress bar + scroll persistence ----
  const bar = document.getElementById("progress-bar");
  function updateProgress() {
    const h = document.documentElement;
    const max = h.scrollHeight - h.clientHeight;
    const pct = max > 0 ? (h.scrollTop / max * 100) : 0;
    bar.style.width = Math.min(100, Math.max(0, pct)) + "%";
  }
  window.addEventListener("scroll", () => {
    updateProgress();
    saveScroll();
  }, { passive: true });

  // ---- Restore prefs ----
  const savedTheme = localStorage.getItem("theme");
  if (savedTheme) {
    document.documentElement.dataset.theme = savedTheme;
    document.getElementById("hljs-light").disabled = (savedTheme === "dark");
    document.getElementById("hljs-dark").disabled  = (savedTheme === "light");
  }
  const savedFont = localStorage.getItem("fontsize");
  if (savedFont) document.documentElement.dataset.fontsize = savedFont;

  // ---- Init ----
  renderNav();
  const initial = (location.hash || "#readme").replace(/^#/, "");
  load(NAV.find(n => n.id === initial) ? initial : "readme", false);
  window.addEventListener("hashchange", () => {
    const id = location.hash.replace(/^#/, "");
    if (NAV.find(n => n.id === id) && id !== activeId) load(id, false);
  });
})();
