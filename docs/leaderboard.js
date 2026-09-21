/* Solve Vision with Code — leaderboard page (leaderboard.html). Vanilla JS, ES2018, no build step, defer-safe.
   Moved here from the old single-page app.js: the sortable 37-row table, the ID→OOD dumbbell SVG, the
   categories table and the efficiency table. Same data, same numbers.

   Data contract (docs/build_site.py writes it; paths relative to this page):
     data/leaderboard.json  {rows:[{model, kind, overall, in_domain, out_of_domain, produced}],   // sorted by overall
                             categories:[{model, Abstraction, Perception, Spatiality, Transformation, Knowledge, overall}],
                             efficiency:[{model, score, produced, attempts, tool_use_rate, tool_calls_per_attempt, timeouts,
                                          median_seconds, mean_input_tokens, mean_output_tokens, ...}]}
     (data/tasks.json is not loaded on this page — DESIGN.md §4; the per-task heatmap is retired, the grid + panel on index.html is the per-task view)
   A bare array is accepted where an object wrapper is described.

   DOM contract (leaderboard.html) — ids this file looks up; each block is skipped when its id is absent:
     #bn-agent #bn-agent-m #bn-video #bn-video-m   hero big numbers (same hero as the home page)
     #lb                                           <table> with <thead> th.sortable[data-k] (rank|model|agent|overall|in_domain|out_of_domain|videos) and an empty <tbody>
     #dumbbell inside #dumbbell-card               <svg> whose first child is its <title>; the card's width sizes the drawing
     #cats, #eff                                   <table> with empty <thead> and <tbody>; a button.showall[data-for="cats"|"eff"] next to each
     #copybib + #bib                               "Copy" button for the cite block (only when the page has one)
   Classes this file creates: tr.kind-agent|open|video .mname .dot .t .k .n .rank .agentcol .hide-m .bar .trk .fil
     svg: .ax .axl .row .lbl .stem .narrow ; .status .status.err */
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
var SVG = 'http://www.w3.org/2000/svg';
function svg(tag, attrs) {
  var n = document.createElementNS(SVG, tag), k;
  attrs = attrs || {};
  for (k in attrs) if (attrs[k] != null) n.setAttribute(k, attrs[k]);
  for (var i = 2; i < arguments.length; i++) appendKids(n, arguments[i]);
  return n;
}
function f3(v) { return (v == null || v === '' || isNaN(+v)) ? '—' : (+v).toFixed(3); }
function f2(v) { return (v == null || v === '' || isNaN(+v)) ? '—' : (+v).toFixed(2); }
function f1(v) { return (v == null || v === '' || isNaN(+v)) ? '—' : (+v).toFixed(1); }
function fint(v) { return (v == null || v === '' || isNaN(+v)) ? '—' : Math.round(+v).toLocaleString('en-US'); }
function num(v) { return (v === '' || v == null || isNaN(+v)) ? null : +v; }
function kindOf(k) {
  k = String(k && k.kind != null ? k.kind : (k || '')).toLowerCase();
  return k.indexOf('video') >= 0 ? 'video' : k.indexOf('open') >= 0 ? 'open' : 'agent';
}
function loadJSON(path) {
  return fetch(path, { cache: 'no-cache' }).then(function (r) {
    if (!r.ok) throw new Error(path + ': HTTP ' + r.status);
    return r.json();
  });
}
function statusNode(msg, err) { return el('div', { class: 'status' + (err ? ' err' : '') }, msg); }
var COLORS = { agent: '#2997ff', open: '#5ac8fa', video: '#8e8e93' };
var KIND_LABEL = { agent: 'coding agent', open: 'open weights', video: 'video model' };

/* ----------------------------------------------------------------- hero */
function renderHero(rows) {
  var best = function (k) {
    return rows.filter(function (r) { return r._kind === k && num(r.overall) != null; })
      .sort(function (a, b) { return b.overall - a.overall; })[0];
  };
  var a = best('agent'), v = best('video'), put = function (id, txt) { var n = document.getElementById(id); if (n) n.textContent = txt; };
  if (a) { put('bn-agent', f3(a.overall)); put('bn-agent-m', a.model); }
  if (v) { put('bn-video', f3(v.overall)); put('bn-video-m', v.model); }
}

/* ------------------------------------------------------------ leaderboard */
/* agent column: the harness version behind each closed-model row (the paper's setup); open-weight rows ran under OpenCode */
var AGENT_VERSION = { 'Codex': 'Codex CLI 0.154.0', 'Claude Code': 'Claude Code 2.1.268', 'Gemini CLI': 'Gemini CLI 0.59.0' };
function agentOf(r, kind) {
  if (r.agent) return r.agent;
  if (kind === 'video') return null;
  if (kind === 'open') return 'OpenCode 1.18.30';
  return AGENT_VERSION[String(r.model).split('/')[0].trim()] || null;
}
function prepareLeaderboard(rows, eff) {
  var produced = new Map((eff || []).map(function (e) { return [e.model, num(e.produced)]; }));
  return rows.map(function (r, i) {
    var _kind = kindOf(r), videos = null;
    if (_kind !== 'video') { videos = num(r.videos); if (videos == null) videos = produced.has(r.model) ? produced.get(r.model) : num(r.produced); }
    return Object.assign({}, r, { rank: num(r.rank) != null ? num(r.rank) : i + 1, _kind: _kind, videos: videos, agent: agentOf(r, _kind) });
  });
}
function renderLeaderboard(rows) {
  var table = $('#lb'); if (!table) return;
  var tbody = table.tBodies[0] || table.appendChild(el('tbody'));
  var heads = Array.prototype.slice.call(table.querySelectorAll('th.sortable'));
  var sort = { k: 'rank', dir: 1 };
  var draw = function () {
    var sorted = rows.slice().sort(function (a, b) {
      var x = a[sort.k], y = b[sort.k];
      if (x == null && y == null) return 0;
      if (x == null) return 1; if (y == null) return -1;
      if (typeof x === 'string') return sort.dir * x.localeCompare(y);
      return sort.dir * (x - y);
    });
    setKids(tbody, sorted.map(function (r) {
      return el('tr', { class: 'kind-' + r._kind },
        el('td', { class: 'n rank' }, r.rank),
        el('td', {}, el('div', { class: 'mname' }, el('span', { class: 'dot', 'aria-hidden': 'true' }),
          el('span', { class: 't', title: r.model }, r.model, el('span', { class: 'k' }, KIND_LABEL[r._kind])))),
        el('td', { class: 'agentcol hide-m' }, r.agent || '—'),
        el('td', { class: 'n' }, el('span', { class: 'bar' },
          el('span', { class: 'trk', 'aria-hidden': 'true' }, el('span', { class: 'fil', style: 'width:' + (Math.max(0, Math.min(1, r.overall || 0)) * 100) + '%' })),
          el('b', {}, f3(r.overall)))),
        el('td', { class: 'n' }, f3(r.in_domain)),
        el('td', { class: 'n' }, f3(r.out_of_domain)),
        el('td', { class: 'n hide-m' }, r.videos == null ? '—' : r.videos));
    }));
    heads.forEach(function (h) {
      var on = h.dataset.k === sort.k;
      h.classList.toggle('asc', on && sort.dir === 1); h.classList.toggle('desc', on && sort.dir === -1);
      h.setAttribute('aria-sort', on ? (sort.dir === 1 ? 'ascending' : 'descending') : 'none');
    });
  };
  var toggle = function (k) { sort = sort.k === k ? { k: k, dir: -sort.dir } : { k: k, dir: ['rank', 'model', 'agent'].indexOf(k) >= 0 ? 1 : -1 }; draw(); };
  heads.forEach(function (h) {
    if (!h.hasAttribute('tabindex')) h.setAttribute('tabindex', '0');
    h.addEventListener('click', function () { toggle(h.dataset.k); });
    h.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(h.dataset.k); } });
  });
  draw();
}

/* --------------------------------------------------------------- dumbbell */
function renderDumbbell(rows) {
  var node = $('#dumbbell'); if (!node) return;
  var card = $('#dumbbell-card') || node.parentNode;
  var data = rows.filter(function (r) { return r._kind === 'video' || r.videos == null || r.videos > 0; })
    .filter(function (r) { return r.in_domain != null && r.out_of_domain != null; })
    .sort(function (a, b) { return b.overall - a.overall; });
  var title = node.firstElementChild && node.firstElementChild.tagName.toLowerCase() === 'title' ? node.firstElementChild : null;
  var draw = function () {
    var W = Math.max(320, card.clientWidth - 32), narrow = W < 640;
    var L = narrow ? 132 : 250, R = 24, maxCh = Math.floor((L - 14) / (narrow ? 6.3 : 6.9)), T = 26, rowH = narrow ? 22 : 24, H = T + data.length * rowH + 14;
    var x = function (v) { return L + v * (W - L - R); };
    node.setAttribute('viewBox', '0 0 ' + W + ' ' + H); node.setAttribute('width', W); node.setAttribute('height', H); node.classList.toggle('narrow', narrow);
    var kids = [];
    for (var t = 0; t <= 1.0001; t += 0.25) {
      kids.push(svg('line', { class: 'ax', x1: x(t), x2: x(t), y1: T - 8, y2: H - 10 }));
      kids.push(svg('text', { class: 'axl', x: x(t), y: T - 12, 'text-anchor': 'middle' }, t.toFixed(2)));
    }
    data.forEach(function (r, i) {
      var y = T + i * rowH + rowH / 2, c = COLORS[r._kind];
      var label = r.model.length > maxCh ? r.model.slice(0, maxCh - 1) + '…' : r.model;
      kids.push(svg('g', { class: 'row', tabindex: '0' },
        svg('title', {}, r.model + '\nIn-Domain ' + f3(r.in_domain) + ' → Out-of-Domain ' + f3(r.out_of_domain)),
        svg('rect', { x: 0, y: y - rowH / 2, width: W, height: rowH, fill: 'transparent' }),
        svg('text', { class: 'lbl', x: L - 10, y: y + 4, 'text-anchor': 'end' }, label),
        svg('line', { class: 'stem', x1: x(r.in_domain), x2: x(r.out_of_domain), y1: y, y2: y, stroke: c, 'stroke-width': 2, 'stroke-opacity': 0.7, 'stroke-linecap': 'round' }),
        svg('circle', { cx: x(r.in_domain), cy: y, r: 4.5, fill: '#000', stroke: c, 'stroke-width': 2 }),
        svg('circle', { cx: x(r.out_of_domain), cy: y, r: 5, fill: c, stroke: '#000', 'stroke-width': 2 })));
    });
    setKids(node, title, kids);
  };
  draw();
  var t; addEventListener('resize', function () { clearTimeout(t); t = setTimeout(draw, 120); });
}

/* ------------------------------------------------- categories & efficiency */
function renderSmallTable(id, rows, cols, defaultShow) {
  var table = document.getElementById(id); if (!table) return;
  defaultShow = defaultShow || 6;
  var btn = $('.showall[data-for="' + id + '"]');
  var thead = table.tHead || table.createTHead(), tbody = table.tBodies[0] || table.appendChild(el('tbody'));
  setKids(thead, el('tr', {}, cols.map(function (c) { return el('th', { class: c.n ? 'n' : '', scope: 'col' }, c.label); })));
  setKids(tbody, rows.map(function (r, i) {
    return el('tr', { hidden: i >= defaultShow }, cols.map(function (c) { return el('td', { class: c.n ? 'n' : '' }, c.fmt ? c.fmt(r[c.k], r) : (r[c.k] == null ? '—' : r[c.k])); }));
  }));
  if (!btn) { Array.prototype.forEach.call(tbody.rows, function (tr) { tr.hidden = false; }); return; }
  var open = false;
  if (rows.length <= defaultShow) { btn.hidden = true; return; }
  btn.addEventListener('click', function () {
    open = !open;
    Array.prototype.forEach.call(tbody.rows, function (tr, i) { tr.hidden = !open && i >= defaultShow; });
    btn.textContent = open ? 'Show fewer' : 'Show all agents';
  });
}
function renderCategories(cats) {
  var table = document.getElementById('cats'); if (!table) return;
  if (!cats || !cats.length) { setKids(table.tBodies[0] || table, statusNode('No category table.')); return; }
  var keys = Object.keys(cats[0]).filter(function (k) { return k !== 'model'; });
  var rows = cats.slice().sort(function (a, b) { return (b.overall || 0) - (a.overall || 0); });
  renderSmallTable('cats', rows, [{ k: 'model', label: 'Model' }].concat(keys.map(function (k) { return { k: k, label: k === 'overall' ? 'Overall' : k.slice(0, 5), n: true, fmt: f3 }; })));
}
function renderEfficiency(eff) {
  var table = document.getElementById('eff'); if (!table) return;
  if (!eff || !eff.length) { setKids(table.tBodies[0] || table, statusNode('No efficiency table.')); return; }
  var rows = eff.slice().sort(function (a, b) { return (b.score || 0) - (a.score || 0); });
  renderSmallTable('eff', rows, [
    { k: 'model', label: 'Model' }, { k: 'score', label: 'Score', n: true, fmt: f3 },
    { k: 'produced', label: 'Videos', n: true, fmt: function (v, r) { return fint(v) + (r.attempts ? '/' + fint(r.attempts) : ''); } },
    { k: 'tool_use_rate', label: 'Tool use', n: true, fmt: f2 }, { k: 'tool_calls_per_attempt', label: 'Calls', n: true, fmt: f1 },
    { k: 'timeouts', label: 'Timeouts', n: true, fmt: fint }, { k: 'median_seconds', label: 'Median s', n: true, fmt: fint },
    { k: 'mean_input_tokens', label: 'In tok', n: true, fmt: fint }, { k: 'mean_output_tokens', label: 'Out tok', n: true, fmt: fint }]);
}

/* ------------------------------------------------------------------ boot */
function boot() {
  var copy = $('#copybib'), bib = $('#bib');
  if (copy && bib) copy.addEventListener('click', function () {
    var txt = bib.textContent.replace(/^\s*Copy/, '').trim();
    var done = function (m) { copy.textContent = m; setTimeout(function () { copy.textContent = 'Copy'; }, 1600); };
    if (navigator.clipboard) navigator.clipboard.writeText(txt).then(function () { done('Copied'); }, function () { done('Select & copy'); });
    else done('Select & copy');
  });
  loadJSON('data/leaderboard.json').then(function (lbFile) {
    var rowsRaw = Array.isArray(lbFile) ? lbFile : (lbFile.rows || []);
    var cats = Array.isArray(lbFile) ? null : lbFile.categories, eff = Array.isArray(lbFile) ? null : lbFile.efficiency;
    var catsP = cats ? Promise.resolve(cats) : loadJSON('data/categories.json').catch(function () { return []; });
    var effP = eff ? Promise.resolve(eff) : loadJSON('data/efficiency.json').catch(function () { return []; });
    return effP.then(function (effRows) {
      var lb = prepareLeaderboard(rowsRaw, effRows);
      renderHero(lb); renderLeaderboard(lb); renderDumbbell(lb); renderEfficiency(effRows);
      return catsP.then(renderCategories);
    });
  }).catch(function (err) {
    var table = $('#lb'), tb = table && (table.tBodies[0] || table);
    if (tb) setKids(tb, statusNode('Could not load data/leaderboard.json (' + err.message + ')', true));
  });
}
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot); else boot();
})();
