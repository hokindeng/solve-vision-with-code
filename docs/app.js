/* Solve Vision with Code — home page (index.html). Vanilla JS, ES2018, no build step, defer-safe.
   Redesign in the object-permanence.world shape: hero numbers, filter pills, a 100-tile grid of
   ground-truth videos, and one inline comparison panel (.detail) inserted after the last tile of
   the clicked tile's row (docs/DESIGN.md §3; reference mechanics ported from object-permanence.world/app.py).

   Data contract (docs/build_site.py writes it; paths relative to this page, so the site also serves under a sub-path):
     data/leaderboard.json  {rows:[{model, kind, overall, in_domain, out_of_domain, produced}], categories:[..], efficiency:[..]}
     data/tasks.json        {generated, lanes:{<lane>:{label, kind:"agent"|"agent-open", run}}, closed_lanes:[..], open_lanes:[..],
                             tasks:[{id, short, split:"In-Domain_50"|"Out-of-Domain_50", category, prompt, mean:{<closed lane>:score},
                                     gt:{first_frame, last_frame, video},
                                     instances:[{idx:"00000", prompt, gt:{...}, lanes:{<closed lane>:{score, video, last_frame, solve_url}}}],
                                     open_examples:[{lane, idx, score, video, last_frame}]}]}
   A bare array is accepted where an object wrapper is described.

   DOM contract (index.html carries the same list) — ids this file looks up; every one is optional except #grid:
     #bn-agent #bn-agent-m #bn-video #bn-video-m   hero big numbers (value + model name)
     #filters                                      empty container; this file fills it with .fgroup > button.fpill[data-k][data-v],
                                                   .fgroup.fq > input.fsearch#f-q + span.fcnt#f-cnt
     #grid                                         div.grid; receives div.tile#task-<short> nodes and the single div.detail panel
     #copybib + #bib                               "Copy" button for the cite block
   URL hash: #task-<short> deep-links a task (opens its panel); #split=…&cat=…&q=… mirrors the filter state.
   Classes this file creates (style.css owns their look): .tile .tile-badge .tile-name .sel
     .detail .dtop .dtitle .dmeta .dsp .pbtn .dclose .pinst .mprompt(.clamp) .more .sec .sec.tog(.open) .chev
     .sbar .strack .sthumb(.drag) .shint(.done) .arr .cmpwrap .cmp .cst .chead(.ref) .crow .ccell(.ref) .cmeta .cscore(.top) .clink(.on)
     .vph .code .ch .obody .ogrid .ocard .om .status(.err) .fgroup(.fq) .fpill(.on) .fsearch .fcnt */
(function () {
'use strict';

/* ------------------------------------------------------------- helpers */
function $(s, r) { return (r || document).querySelector(s); }
function el(tag, attrs) {
  var n = document.createElement(tag), k;
  attrs = attrs || {};
  for (k in attrs) {
    var v = attrs[k];
    if (v == null || v === false) continue;
    if (k === 'class') n.className = v;
    else if (k.indexOf('on') === 0) n.addEventListener(k.slice(2), v);
    else if (k === 'dataset') Object.assign(n.dataset, v);
    else n.setAttribute(k, v === true ? '' : v);
  }
  for (var i = 2; i < arguments.length; i++) appendKids(n, arguments[i]);
  return n;
}
function appendKids(n, kid) {
  if (kid == null || kid === false) return;
  if (Array.isArray(kid)) { for (var i = 0; i < kid.length; i++) appendKids(n, kid[i]); return; }
  n.appendChild(kid.nodeType ? kid : document.createTextNode(String(kid)));
}
function setKids(n) { n.textContent = ''; for (var i = 1; i < arguments.length; i++) appendKids(n, arguments[i]); }
function f3(v) { return (v == null || v === '' || isNaN(+v)) ? '—' : (+v).toFixed(3); }
function num(v) { return (v === '' || v == null || isNaN(+v)) ? null : +v; }
function kindOf(k) {
  k = String(k && k.kind != null ? k.kind : (k || '')).toLowerCase();
  return k.indexOf('video') >= 0 ? 'video' : k.indexOf('open') >= 0 ? 'open' : 'agent';
}
function splitOf(s) { s = String(s || ''); return /^out/i.test(s) ? 'Out-of-Domain' : /^in/i.test(s) ? 'In-Domain' : s; }
function splitShort(s) { s = splitOf(s); return s === 'In-Domain' ? 'ID' : s === 'Out-of-Domain' ? 'OOD' : s; }
function normId(s) { return String(s == null ? '' : s).replace(/_data-generator$/, ''); }
function idxNum(i) { var n = parseInt(i, 10); return isNaN(n) ? String(i) : n; }
/* "G-131_select_next_figure_increasing_size_sequence" → ["G-131", "select next figure increasing size sequence"] */
function titleOf(t) {
  var s = normId(t.short || t.id), m = /^([GO]-\d+)_(.+)$/.exec(s);
  return m ? [m[1], m[2].replace(/_/g, ' ')] : [s, ''];
}
function loadJSON(path) {
  return fetch(path, { cache: 'no-cache' }).then(function (r) {
    if (!r.ok) throw new Error(path + ': HTTP ' + r.status);
    return r.json();
  });
}
var textCache = new Map();
function fetchText(path) {
  if (!textCache.has(path)) textCache.set(path, fetch(path).then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.text(); }));
  return textCache.get(path);
}
function statusNode(msg, err) { return el('div', { class: 'status' + (err ? ' err' : '') }, msg); }
function onActivate(node, fn) {
  node.addEventListener('click', fn);
  node.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fn(e); } });
}

/* ---------------------------------------------------------------- lanes */
var LANES = {}, CLOSED = [];
/* "Codex / gpt-6-astra" → agent "Codex", model "gpt-6-astra"; "GLM-5 (OpenCode)" → agent "OpenCode", model "GLM-5" */
function laneParts(l) {
  var label = (LANES[l] && LANES[l].label) || l, m = /^(.*?) \(OpenCode\)$/.exec(label);
  if (m) return { agent: 'OpenCode', model: m[1] };
  var p = label.split(' / ');
  return p.length > 1 ? { agent: p[0], model: p.slice(1).join(' / ') } : { agent: label, model: '' };
}
function laneName(l) { var p = laneParts(l); return p.agent === 'OpenCode' ? p.model : p.agent; }
function laneFull(l) { return (LANES[l] && LANES[l].label) || l; }
function closedLanes(t) {
  if (CLOSED.length) return CLOSED;
  var i = t.instances && t.instances[0];
  return i && i.lanes ? Object.keys(i.lanes) : [];
}
function meanOf(t, l) {
  if (t.mean && t.mean[l] != null) return num(t.mean[l]);
  var s = (t.instances || []).map(function (i) { return num(i.lanes && i.lanes[l] && i.lanes[l].score); }).filter(function (v) { return v != null; });
  return s.length ? s.reduce(function (a, b) { return a + b; }, 0) / s.length : null;
}

/* ----------------------------------------------------------------- hero */
function renderHero(rows) {
  var best = function (k) {
    return rows.filter(function (r) { return kindOf(r) === k && num(r.overall) != null; })
      .sort(function (a, b) { return b.overall - a.overall; })[0];
  };
  var a = best('agent'), v = best('video'), put = function (id, txt) { var n = document.getElementById(id); if (n) n.textContent = txt; };
  if (a) { put('bn-agent', f3(a.overall)); put('bn-agent-m', a.model); }
  if (v) { put('bn-video', f3(v.overall)); put('bn-video-m', v.model); }
}

/* --------------------------------------------------------------- videos */
/* one observer for every video on the page: play when near the viewport, pause when away */
var io = new IntersectionObserver(function (es) {
  es.forEach(function (e) {
    var v = e.target;
    if (e.isIntersecting) { if (!v._paused) { var p = v.play(); if (p && p.catch) p.catch(function () {}); } }
    else v.pause();
  });
}, { rootMargin: '250px 0px' });

function vid(src, opts) {
  opts = opts || {};
  var v = el('video', { muted: true, loop: true, playsinline: true, preload: 'none', poster: opts.poster || null, controls: !!opts.controls, 'aria-label': opts.label || null });
  v.muted = true; v.loop = true; v.playsInline = true;
  v.src = src;
  return v;
}
/* a table cell: the video, or a dashed placeholder — never a blank */
function vcell(src, opts) {
  if (!src) return el('div', { class: 'vph' }, 'no video');
  var v = vid(src, opts);
  v.addEventListener('error', function () {
    io.unobserve(v);
    if (v.parentNode) v.parentNode.replaceChild(el('div', { class: 'vph' }, 'no video'), v);
  }, { once: true });
  return v;
}
function stopVideos(root) {
  var vs = root.querySelectorAll('video');
  for (var i = 0; i < vs.length; i++) { io.unobserve(vs[i]); try { vs[i].pause(); } catch (e) {} }
}
function watchVideos(root) {
  var vs = root.querySelectorAll('video');
  for (var i = 0; i < vs.length; i++) io.observe(vs[i]);
}

/* ---------------------------------------------------------------- tiles */
var grid, TASKS = [], tiles = new Map(), shown = [], openId = null, panel = null, paused = false;

function makeTile(t) {
  var tt = titleOf(t), short = normId(t.short || t.id), gt = t.gt || (t.instances && t.instances[0] && t.instances[0].gt) || {};
  var d = el('div', { class: 'tile', id: 'task-' + short, role: 'button', tabindex: '0', dataset: { id: t.id },
    'aria-label': tt[0] + ' ' + tt[1] + ', ' + splitOf(t.split) + (t.category ? ', ' + t.category : '') });
  d.appendChild(vid(gt.video || '', { poster: gt.last_frame || gt.first_frame, label: 'ground truth ' + short }));
  d.appendChild(el('span', { class: 'tile-badge' }, el('b', {}, splitShort(t.split)), t.category ? ' · ' + t.category : ''));
  d.appendChild(el('div', { class: 'tile-name' }, el('b', {}, tt[0]), tt[1]));
  onActivate(d, function () { toggleTask(t.id); });
  return d;
}

/* the last tile that shares the clicked tile's row (same offsetTop) — the panel goes right after it */
function rowEnd(id) {
  var i = shown.findIndex(function (t) { return t.id === id; });
  if (i < 0) return null;
  var top = tiles.get(id).offsetTop, j = i;
  while (j + 1 < shown.length && tiles.get(shown[j + 1].id).offsetTop === top) j++;
  return tiles.get(shown[j].id);
}
function placePanel() {
  if (!panel || openId == null) return;
  var a = rowEnd(openId);
  if (panel.parentNode) panel.parentNode.removeChild(panel);
  if (a) a.parentNode.insertBefore(panel, a.nextSibling);
}

/* ---------------------------------------------------------------- panel */
function closePanel() {
  if (!panel) return;
  stopVideos(panel);
  if (panel.parentNode) panel.parentNode.removeChild(panel);
  panel = null;
  if (openId != null && tiles.has(openId)) tiles.get(openId).classList.remove('sel');
  openId = null;
}

function toggleTask(id, scroll) {
  if (openId === id) { closePanel(); return; }
  closePanel();
  var t = TASKS.find(function (x) { return x.id === id; }), anchor = rowEnd(id);
  if (!t || !anchor) return;
  paused = false;
  panel = buildPanel(t);
  anchor.parentNode.insertBefore(panel, anchor.nextSibling);
  openId = id; tiles.get(id).classList.add('sel');
  watchVideos(panel);
  initScroller(panel);
  if (scroll !== false) panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function buildPanel(t) {
  var lanes = closedLanes(t), tt = titleOf(t), ins = t.instances || [];
  var p = el('div', { class: 'detail', role: 'region', 'aria-label': 'comparison for ' + tt[0] });

  /* top bar */
  var tgl = el('button', { class: 'pbtn', type: 'button', onclick: function () { toggleAll(tgl); } }, '❚❚ Pause all');
  p.appendChild(el('div', { class: 'dtop' },
    el('span', { class: 'dtitle' }, el('b', {}, tt[0]), tt[1]),
    el('span', { class: 'dmeta' }, splitOf(t.split) + (t.category ? ' · ' + t.category : '')),
    el('span', { class: 'dsp' }),
    el('button', { class: 'pbtn', type: 'button', onclick: function () { replayAll(tgl); } }, '⟲ Replay all'),
    tgl,
    el('button', { class: 'dclose', type: 'button', 'aria-label': 'close', onclick: closePanel }, '✕')));

  /* prompt: instance 0's, a tiny switch when the five prompts differ, clamped when long */
  var prompts = ins.map(function (i) { return i.prompt || t.prompt || ''; });
  if (!prompts.length) prompts = [t.prompt || ''];
  var prompt = el('div', { class: 'mprompt' }), more = null, expanded = false;
  var CLAMP = 'display:-webkit-box;-webkit-line-clamp:4;-webkit-box-orient:vertical;overflow:hidden';
  var clamp = function (on) { prompt.classList.toggle('clamp', on); prompt.setAttribute('style', on ? CLAMP : ''); };
  var showPrompt = function (s) {
    var long = s.length > 420;
    setKids(prompt, s);
    clamp(long && !expanded);
    if (more) more.hidden = !long;
  };
  if (prompts.some(function (s) { return s.length > 420; })) {
    more = el('button', { class: 'fpill more', type: 'button', 'aria-expanded': 'false', onclick: function () {
      expanded = !expanded; clamp(!expanded);
      more.textContent = expanded ? 'Less' : 'Full prompt'; more.setAttribute('aria-expanded', expanded ? 'true' : 'false');
    } }, 'Full prompt');
  }
  if (prompts.some(function (s) { return s !== prompts[0]; })) {
    var btns = prompts.map(function (s, k) {
      return el('button', { class: 'fpill' + (k === 0 ? ' on' : ''), type: 'button', 'aria-pressed': k === 0 ? 'true' : 'false', onclick: function () {
        showPrompt(s);
        btns.forEach(function (b, i) { b.classList.toggle('on', i === k); b.setAttribute('aria-pressed', i === k ? 'true' : 'false'); });
      } }, idxNum(ins[k] && ins[k].idx != null ? ins[k].idx : k));
    });
    p.appendChild(el('div', { class: 'pinst', role: 'group', 'aria-label': 'prompt of instance' }, el('span', {}, 'prompt of instance'), btns));
  }
  showPrompt(prompts[0]);
  p.appendChild(prompt);
  if (more) p.appendChild(el('div', { class: 'pinst' }, more));

  /* comparison table: sticky first column (instance first frame), one column per agent, ground truth last */
  p.appendChild(el('div', { class: 'sec' }, 'Comparison', el('span', {}, ins.length + ' instances × ' + lanes.length + ' agents · official evaluator score under each video')));
  p.appendChild(el('div', { class: 'sbar' }, el('div', { class: 'strack' }, el('div', { class: 'sthumb' })), el('span', { class: 'shint' }, 'Scroll right ', el('span', { class: 'arr' }, '→'))));
  var cmp = el('div', { class: 'cmp', style: 'grid-template-columns:96px repeat(' + (lanes.length + 1) + ',minmax(250px,1fr))' });
  cmp.appendChild(el('div', { class: 'cst' }));
  lanes.forEach(function (l) {
    var lp = laneParts(l), mean = meanOf(t, l);
    cmp.appendChild(el('div', { class: 'chead', title: laneFull(l) }, el('b', {}, lp.agent), lp.model ? el('small', {}, lp.model) : null, mean != null ? el('em', {}, 'mean ' + f3(mean)) : null));
  });
  cmp.appendChild(el('div', { class: 'chead ref' }, el('b', {}, 'Ground truth'), el('small', {}, 'official video')));
  var wrap = el('div', { class: 'cmpwrap' }), codeBox = null, codeKey = null;
  var toggleCode = function (key, path, btn) {
    p.querySelectorAll('.clink').forEach(function (b) { b.classList.remove('on'); b.setAttribute('aria-expanded', 'false'); });
    if (codeBox) { codeBox.parentNode.removeChild(codeBox); codeBox = null; }
    if (codeKey === key) { codeKey = null; return; }
    codeKey = key;
    var pre = el('pre', { style: 'margin:0;padding:12px 14px;overflow:auto;max-height:440px;font-size:12px;line-height:1.5;white-space:pre;tab-size:4' }, 'Loading…');
    codeBox = el('div', { class: 'code', style: 'margin-top:14px;border:1px solid var(--border);border-radius:10px;background:#0c0c0e;min-width:0' },
      el('div', { class: 'ch', style: 'display:flex;justify-content:space-between;gap:12px;padding:8px 14px;border-bottom:1px solid var(--border);font-size:12px;color:var(--t2)' },
        el('span', {}, el('b', {}, key.split('|')[0]), ' · ' + path.split('/').slice(-3).join('/')),
        el('a', { href: path, target: '_blank', rel: 'noopener' }, 'raw')), pre);
    wrap.parentNode.insertBefore(codeBox, wrap.nextSibling);
    btn.classList.add('on'); btn.setAttribute('aria-expanded', 'true');
    fetchText(path).then(function (txt) { pre.textContent = txt; }, function (e) { setKids(pre, el('span', { class: 'err' }, 'could not load ' + path + ' (' + e.message + ')')); });
  };
  ins.forEach(function (i, k) {
    var label = idxNum(i.idx != null ? i.idx : k), gt = i.gt || t.gt || {};
    cmp.appendChild(el('div', { class: 'crow cst' }, el('div', { style: 'display:flex;flex-direction:column;align-items:center;gap:6px' },
      gt.first_frame ? el('img', { src: gt.first_frame, alt: 'first frame of instance ' + label, loading: 'lazy', decoding: 'async', style: 'width:72px;height:72px;object-fit:cover;border-radius:6px;display:block;background:#000' }) : null,
      el('span', {}, 'instance ' + label))));
    lanes.forEach(function (l) {
      var d = (i.lanes && i.lanes[l]) || {}, solve = d.solve_url || d.solve, score = d.score == null ? (d.video ? null : 0) : num(d.score);
      var key = laneParts(l).agent + '|' + label, btn = solve ? el('button', { class: 'clink', type: 'button', 'aria-expanded': 'false' }, 'program') : null;
      if (btn) btn.addEventListener('click', function () { toggleCode(key, solve, btn); });
      cmp.appendChild(el('div', { class: 'ccell' },
        vcell(d.video, { poster: d.last_frame, controls: true, label: laneFull(l) + ' instance ' + label }),
        el('div', { class: 'cmeta' },
          el('span', { class: 'cscore' + (score != null && score >= 0.9995 ? ' top' : ''), style: score != null && score >= 0.9995 ? 'color:var(--accent)' : null }, score == null ? 'no score' : f3(score)),
          btn)));
    });
    cmp.appendChild(el('div', { class: 'ccell ref' },
      vcell(gt.video, { poster: gt.last_frame, controls: true, label: 'ground truth instance ' + label }),
      el('div', { class: 'cmeta' }, el('span', { class: 'cscore' }, 'ground truth'))));
  });
  wrap.appendChild(cmp);
  p.appendChild(wrap);

  /* collapsed open-weight examples of this task */
  var ex = (t.open_examples || []).filter(function (o) { return o.video; });
  if (ex.length) {
    var names = []; ex.forEach(function (o) { var n = laneName(o.lane); if (names.indexOf(n) < 0) names.push(n); });
    var body = el('div', { class: 'obody' }), open = false;
    var head = el('div', { class: 'sec tog', role: 'button', tabindex: '0', 'aria-expanded': 'false' },
      el('span', { class: 'chev' }, '▸'), 'Open-weight examples',
      el('span', {}, ex.length + ' videos · ' + names.join(', ') + ' through OpenCode'));
    onActivate(head, function () {
      open = !open;
      head.classList.toggle('open', open); head.setAttribute('aria-expanded', open ? 'true' : 'false');
      stopVideos(body);
      if (!open) { setKids(body); return; }
      setKids(body, el('div', { class: 'ogrid' }, ex.map(function (o) { return openCard(o, t); })));
      watchVideos(body);
      if (paused) body.querySelectorAll('video').forEach(function (v) { v._paused = true; v.pause(); });
    });
    p.appendChild(head); p.appendChild(body);
  }
  return p;
}

function openCard(o, t) {
  return el('div', { class: 'ocard' },
    vcell(o.video, { poster: o.last_frame, controls: true, label: laneFull(o.lane) + ' ' + titleOf(t)[0] + ' instance ' + idxNum(o.idx) }),
    el('div', { class: 'om' }, el('b', { title: laneFull(o.lane) }, laneName(o.lane)),
      el('span', {}, 'instance ' + idxNum(o.idx)), el('span', { class: 'cscore' }, f3(o.score))));
}

function toggleAll(btn) {
  if (!panel) return;
  paused = !paused;
  panel.querySelectorAll('video').forEach(function (v) { v._paused = paused; if (paused) v.pause(); else { var p = v.play(); if (p && p.catch) p.catch(function () {}); } });
  btn.textContent = paused ? '▶ Play all' : '❚❚ Pause all';
}
function replayAll(btn) {
  if (!panel) return;
  paused = false;
  panel.querySelectorAll('video').forEach(function (v) { v._paused = false; try { v.currentTime = 0; var p = v.play(); if (p && p.catch) p.catch(function () {}); } catch (e) {} });
  btn.textContent = '❚❚ Pause all';
}

/* custom white scroll bar above the table, mirrored to .cmpwrap's horizontal scroll */
function initScroller(p) {
  var wrap = $('.cmpwrap', p), bar = $('.sbar', p);
  if (!wrap || !bar) return;
  var track = $('.strack', bar), thumb = $('.sthumb', bar), hint = $('.shint', bar);
  var maxScroll = function () { return wrap.scrollWidth - wrap.clientWidth; };
  function layout() {
    var ms = maxScroll();
    bar.style.display = ms <= 1 ? 'none' : '';
    if (ms <= 1) return;
    var tw = track.clientWidth, w = Math.max(48, tw * wrap.clientWidth / wrap.scrollWidth);
    thumb.style.width = w + 'px';
    thumb.style.left = (wrap.scrollLeft / ms) * (tw - w) + 'px';
    hint.classList.toggle('done', wrap.scrollLeft >= ms - 2);
  }
  wrap.addEventListener('scroll', layout, { passive: true });
  if (window.ResizeObserver) new ResizeObserver(layout).observe(wrap); else addEventListener('resize', layout);
  var drag = null;
  thumb.addEventListener('pointerdown', function (e) { drag = { x: e.clientX, sl: wrap.scrollLeft }; thumb.classList.add('drag'); thumb.setPointerCapture(e.pointerId); e.preventDefault(); });
  thumb.addEventListener('pointermove', function (e) {
    if (!drag) return;
    var tw = track.clientWidth, w = thumb.offsetWidth;
    wrap.scrollLeft = drag.sl + (e.clientX - drag.x) * maxScroll() / (tw - w);
  });
  var stop = function () { drag = null; thumb.classList.remove('drag'); };
  thumb.addEventListener('pointerup', stop); thumb.addEventListener('pointercancel', stop);
  track.addEventListener('click', function (e) {
    if (e.target === thumb) return;
    var r = track.getBoundingClientRect(), w = thumb.offsetWidth;
    wrap.scrollTo({ left: ((e.clientX - r.left - w / 2) / (r.width - w)) * maxScroll(), behavior: 'smooth' });
  });
  layout();
  /* videos load lazily, so the row heights settle late: re-measure a few times */
  [200, 800, 2000].forEach(function (ms) { setTimeout(layout, ms); });
}

/* -------------------------------------------------------------- filters */
var FILTER = { split: '', cat: '', q: '' }, HAY = new Map(), PILLS = [];
function renderFilters(tasks) {
  var box = $('#filters');
  if (!box) return;
  var cats = []; tasks.forEach(function (t) { if (t.category && cats.indexOf(t.category) < 0) cats.push(t.category); }); cats.sort();
  var count = function (pred) { return tasks.filter(pred).length; };
  var group = function (name, key, items) {
    var g = el('div', { class: 'fgroup', role: 'group', 'aria-label': name });
    items.forEach(function (it) {
      var b = el('button', { class: 'fpill', type: 'button', 'aria-pressed': 'false', dataset: { k: key, v: it.v } }, it.label, it.n != null ? el('small', {}, ' ' + it.n) : null);
      b.addEventListener('click', function () { FILTER[key] = it.v; applyFilters(); writeHash(); });
      PILLS.push(b); g.appendChild(b);
    });
    return g;
  };
  var q = el('input', { class: 'fsearch', id: 'f-q', type: 'search', placeholder: 'Search id, name or prompt…', autocomplete: 'off', 'aria-label': 'search tasks' });
  var timer; q.addEventListener('input', function () { clearTimeout(timer); timer = setTimeout(function () { FILTER.q = q.value.trim(); applyFilters(); writeHash(); }, 120); });
  setKids(box,
    group('split', 'split', [{ v: '', label: 'All', n: tasks.length },
      { v: 'In-Domain', label: 'In-Domain', n: count(function (t) { return splitOf(t.split) === 'In-Domain'; }) },
      { v: 'Out-of-Domain', label: 'Out-of-Domain', n: count(function (t) { return splitOf(t.split) === 'Out-of-Domain'; }) }]),
    group('category', 'cat', [{ v: '', label: 'All categories' }].concat(cats.map(function (c) { return { v: c, label: c, n: count(function (t) { return t.category === c; }) }; }))),
    el('div', { class: 'fgroup fq' }, q, el('span', { class: 'fcnt', id: 'f-cnt', 'aria-live': 'polite' })));
}
/* filtering closes an open panel (DESIGN.md §3.1); the pills and the search box are re-synced from FILTER */
function applyFilters() {
  var s = FILTER.split, c = FILTER.cat, q = FILTER.q.toLowerCase();
  closePanel();
  shown = TASKS.filter(function (t) { return (!s || splitOf(t.split) === s) && (!c || t.category === c) && (!q || HAY.get(t.id).indexOf(q) >= 0); });
  setKids(grid, shown.map(function (t) { return tiles.get(t.id); }));
  if (!shown.length) grid.appendChild(statusNode('No task matches.'));
  /* only the tiles on the page are observed; the rest stay paused */
  tiles.forEach(function (tile) { var v = tile.querySelector('video'); if (!v) return; if (tile.isConnected) io.observe(v); else { io.unobserve(v); v.pause(); } });
  PILLS.forEach(function (b) { var on = FILTER[b.dataset.k] === b.dataset.v; b.classList.toggle('on', on); b.setAttribute('aria-pressed', on ? 'true' : 'false'); });
  var qi = $('#f-q'); if (qi && qi.value.trim() !== FILTER.q) qi.value = FILTER.q;
  var cnt = $('#f-cnt'); if (cnt) cnt.textContent = shown.length + ' / ' + TASKS.length + ' tasks';
}
/* URL hash: "#split=In-Domain&cat=Perception&q=…" (only the non-default parts; empty when all default), or "#task-<short>" */
var settingHash = false;
function writeHash() {
  var parts = [];
  if (FILTER.split) parts.push('split=' + encodeURIComponent(FILTER.split));
  if (FILTER.cat) parts.push('cat=' + encodeURIComponent(FILTER.cat));
  if (FILTER.q) parts.push('q=' + encodeURIComponent(FILTER.q));
  var h = parts.length ? '#' + parts.join('&') : '';
  if (h === location.hash || (!h && !location.hash)) return;
  settingHash = true;
  history.replaceState(null, '', location.pathname + location.search + h);
  settingHash = false;
}
function readHash() {
  var h = decodeURIComponent(location.hash.replace(/^#/, ''));
  if (!h) { FILTER = { split: '', cat: '', q: '' }; applyFilters(); return; }
  if (/^task-/.test(h)) { gotoTask(h.slice(5)); return; }
  var f = { split: '', cat: '', q: '' };
  h.split('&').forEach(function (kv) { var i = kv.indexOf('='); if (i < 0) return; var k = kv.slice(0, i), v = kv.slice(i + 1); if (k in f) f[k] = v; });
  FILTER = f; applyFilters();
}
/* open a task by short id or full id, clearing the filters when it is hidden */
function gotoTask(ref) {
  var key = normId(String(ref || '').replace(/^#?task-/, ''));
  var t = TASKS.find(function (x) { return normId(x.short || x.id) === key || normId(x.id) === key; });
  if (!t) return false;
  if (!tiles.get(t.id).isConnected) { FILTER = { split: '', cat: '', q: '' }; applyFilters(); }
  if (openId !== t.id) toggleTask(t.id, false);
  tiles.get(t.id).scrollIntoView({ behavior: 'smooth', block: 'start' });
  return true;
}

/* ----------------------------------------------------------------- boot */
function boot() {
  grid = $('#grid');
  var copy = $('#copybib'), bib = $('#bib');
  if (copy && bib) copy.addEventListener('click', function () {
    var txt = bib.textContent.replace(/^\s*Copy/, '').trim();
    var done = function (s) { copy.textContent = s; setTimeout(function () { copy.textContent = 'Copy'; }, 1600); };
    if (navigator.clipboard) navigator.clipboard.writeText(txt).then(function () { done('Copied'); }, function () { done('Select & copy'); });
    else done('Select & copy');
  });

  loadJSON('data/leaderboard.json').then(function (lb) {
    renderHero(Array.isArray(lb) ? lb : (lb.rows || []));
  }).catch(function (err) { var n = $('#bn-agent-m'); if (n) n.textContent = 'leaderboard unavailable (' + err.message + ')'; });

  if (!grid) return;
  setKids(grid, statusNode('Loading tasks…'));
  loadJSON('data/tasks.json').then(function (tf) {
    TASKS = Array.isArray(tf) ? tf : (tf.tasks || []);
    LANES = (!Array.isArray(tf) && tf.lanes) || {};
    CLOSED = (!Array.isArray(tf) && tf.closed_lanes) || [];
    if (!TASKS.length) { setKids(grid, statusNode('data/tasks.json has no tasks.', true)); return; }
    TASKS.forEach(function (t) {
      tiles.set(t.id, makeTile(t));
      HAY.set(t.id, [t.id, t.short, t.category, splitOf(t.split), splitShort(t.split), t.prompt].concat((t.instances || []).map(function (i) { return i.prompt; })).join(' ').toLowerCase());
    });
    renderFilters(TASKS);
    readHash();
    addEventListener('hashchange', function () { if (!settingHash) readHash(); });
  }).catch(function (err) { setKids(grid, statusNode('Could not load data/tasks.json (' + err.message + ')', true)); });

  document.addEventListener('keydown', function (e) { if (e.key === 'Escape') closePanel(); });
  var rto = null;
  addEventListener('resize', function () { clearTimeout(rto); rto = setTimeout(placePanel, 150); });
}
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot); else boot();
})();
