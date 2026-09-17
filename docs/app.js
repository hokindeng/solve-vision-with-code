/* Solve Vision with Code — results page. Vanilla JS, no build step.
   Data contract (docs/build_site.py writes it; paths relative to this page, so it works under a Pages sub-path):
     data/leaderboard.json  {rows:[{model, kind, overall, in_domain, out_of_domain, produced}],   // table 1, sorted by overall
                             categories:[{model, Abstraction, Perception, Spatiality, Transformation, Knowledge, overall}],
                             efficiency:[{model, score, produced, attempts, tool_use_rate, tool_calls_per_attempt, timeouts,
                                          median_seconds, mean_input_tokens, mean_output_tokens, ...}]}
     data/tasks.json        {generated, lanes:{<lane>:{label, kind:"agent"|"agent-open", run}}, closed_lanes:[..], open_lanes:[..],
                             tasks:[{id, short, split:"In-Domain_50"|"Out-of-Domain_50", category, prompt, mean:{<closed lane>:score},
                                     gt:{first_frame, last_frame, video},
                                     instances:[{idx:"00000", prompt, gt:{...}, lanes:{<closed lane>:{score, video, last_frame, solve_url}}}],
                                     open_examples:[{lane, idx, score, video, last_frame}]}]}
   A bare array is accepted where an object wrapper is described. */
(() => {
'use strict';

const $ = (s, r = document) => r.querySelector(s);
const el = (tag, attrs = {}, ...kids) => {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === 'class') n.className = v;
    else if (k.startsWith('on')) n.addEventListener(k.slice(2), v);
    else if (k === 'dataset') Object.assign(n.dataset, v);
    else n.setAttribute(k, v === true ? '' : v);
  }
  for (const k of kids.flat()) if (k != null) n.append(k.nodeType ? k : document.createTextNode(String(k)));
  return n;
};
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const f3 = v => (v == null || Number.isNaN(+v)) ? '—' : (+v).toFixed(3);
const f2 = v => (v == null || Number.isNaN(+v)) ? '—' : (+v).toFixed(2);
const fint = v => (v == null || Number.isNaN(+v)) ? '—' : Math.round(+v).toLocaleString('en-US');
const num = v => (v === '' || v == null) ? null : (Number.isNaN(+v) ? null : +v);
const SVG = 'http://www.w3.org/2000/svg';
const svg = (tag, attrs = {}, ...kids) => {
  const n = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) if (v != null) n.setAttribute(k, v);
  for (const k of kids.flat()) if (k != null) n.append(k.nodeType ? k : document.createTextNode(String(k)));
  return n;
};
const COLORS = { agent: '#2997ff', open: '#5ac8fa', video: '#8e8e93' };
const KIND_LABEL = { agent: 'coding agent', open: 'open weights', video: 'video model' };

/* kind: "coding agent" | "coding agent (open weights)" | "video model" | lanes[].kind "agent" | "agent-open" */
const kindOf = k => { k = String(k?.kind ?? k ?? '').toLowerCase(); return k.includes('video') ? 'video' : k.includes('open') ? 'open' : 'agent'; };
/* split: "In-Domain_50" | "In_Domain" | "In-Domain" → "In-Domain" / "Out-of-Domain" */
const splitOf = s => /^out/i.test(String(s || '')) ? 'Out-of-Domain' : /^in/i.test(String(s || '')) ? 'In-Domain' : String(s || '');
/* task ids: directory name; table 4 strips the `_data-generator` suffix. */
const normId = s => String(s ?? '').replace(/_data-generator$/, '');
const idxNum = i => { const n = parseInt(i, 10); return Number.isNaN(n) ? String(i) : n; };

async function loadJSON(path) {
  const r = await fetch(path, { cache: 'no-cache' });
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  return r.json();
}
const status = (container, msg, err = false) => { container.replaceChildren(el('div', { class: 'status' + (err ? ' err' : '') }, msg)); };

/* ------------------------------------------------------------------ hero */
function renderHero(lb) {
  const best = k => lb.filter(r => r._kind === k).sort((a, b) => b.overall - a.overall)[0];
  const a = best('agent'), v = best('video');
  if (a) { $('#bn-agent').textContent = f3(a.overall); $('#bn-agent-m').textContent = a.model; }
  if (v) { $('#bn-video').textContent = f3(v.overall); $('#bn-video-m').textContent = v.model; }
}

/* ------------------------------------------------------------ leaderboard */
const CLOSED_AGENT = { 'Codex/gpt-6-astra': 'Codex CLI 0.154.0', 'Claude Code/Fable 5.1': 'Claude Code 2.1.268', 'Gemini CLI/3.1 Pro': 'Gemini CLI 0.59.0' };
function prepareLeaderboard(rows, eff) {
  const produced = new Map(eff.map(e => [e.model, num(e.produced)]));
  return rows.map((r, i) => {
    const _kind = kindOf(r);
    const videos = _kind === 'video' ? null : (num(r.videos) ?? produced.get(r.model) ?? num(r.produced));
    const agent = r.agent || (_kind === 'video' ? null : CLOSED_AGENT[r.model] || (_kind === 'open' ? 'OpenCode 1.18.30' : null));
    return { ...r, rank: num(r.rank) ?? i + 1, _kind, videos, agent };
  });
}
function renderLeaderboard(rows) {
  const tbody = $('#lb tbody');
  const heads = [...document.querySelectorAll('#lb th.sortable')];
  let sort = { k: 'rank', dir: 1 };
  const draw = () => {
    const sorted = [...rows].sort((a, b) => {
      const x = a[sort.k], y = b[sort.k];
      if (x == null && y == null) return 0;
      if (x == null) return 1; if (y == null) return -1;
      if (typeof x === 'string') return sort.dir * x.localeCompare(y);
      return sort.dir * (x - y);
    });
    tbody.replaceChildren(...sorted.map(r => el('tr', { class: 'kind-' + r._kind },
      el('td', { class: 'n rank' }, r.rank),
      el('td', {}, el('div', { class: 'mname' }, el('span', { class: 'dot', 'aria-hidden': 'true' }),
        el('span', { class: 't', title: r.model }, r.model, el('span', { class: 'k' }, KIND_LABEL[r._kind])))),
      el('td', { class: 'agentcol hide-m' }, r.agent || '—'),
      el('td', { class: 'n' }, el('span', { class: 'bar' },
        el('span', { class: 'trk', 'aria-hidden': 'true' }, el('span', { class: 'fil', style: `width:${Math.max(0, Math.min(1, r.overall)) * 100}%` })),
        el('b', {}, f3(r.overall)))),
      el('td', { class: 'n' }, f3(r.in_domain)),
      el('td', { class: 'n' }, f3(r.out_of_domain)),
      el('td', { class: 'n hide-m' }, r.videos == null ? '—' : r.videos))));
    heads.forEach(h => { const on = h.dataset.k === sort.k; h.classList.toggle('asc', on && sort.dir === 1); h.classList.toggle('desc', on && sort.dir === -1); h.setAttribute('aria-sort', on ? (sort.dir === 1 ? 'ascending' : 'descending') : 'none'); });
  };
  const toggle = k => { sort = sort.k === k ? { k, dir: -sort.dir } : { k, dir: ['rank', 'model', 'agent'].includes(k) ? 1 : -1 }; draw(); };
  heads.forEach(h => { h.addEventListener('click', () => toggle(h.dataset.k)); h.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(h.dataset.k); } }); });
  draw();
}

/* --------------------------------------------------------------- dumbbell */
function renderDumbbell(rows) {
  const node = $('#dumbbell'), card = $('#dumbbell-card');
  const data = rows.filter(r => r._kind === 'video' || r.videos == null || r.videos > 0)
    .filter(r => r.in_domain != null && r.out_of_domain != null).sort((a, b) => b.overall - a.overall);
  const draw = () => {
    const W = Math.max(320, card.clientWidth - 32), narrow = W < 640;
    const L = narrow ? 132 : 250, R = 24, maxCh = Math.floor((L - 14) / (narrow ? 6.3 : 6.9)), T = 26, rowH = narrow ? 22 : 24, H = T + data.length * rowH + 14;
    const x = v => L + v * (W - L - R);
    node.setAttribute('viewBox', `0 0 ${W} ${H}`); node.setAttribute('width', W); node.setAttribute('height', H); node.classList.toggle('narrow', narrow);
    const kids = [];
    for (let t = 0; t <= 1.0001; t += 0.25) {
      kids.push(svg('line', { class: 'ax', x1: x(t), x2: x(t), y1: T - 8, y2: H - 10 }));
      kids.push(svg('text', { class: 'axl', x: x(t), y: T - 12, 'text-anchor': 'middle' }, t.toFixed(2)));
    }
    data.forEach((r, i) => {
      const y = T + i * rowH + rowH / 2, c = COLORS[r._kind];
      const label = r.model.length > maxCh ? r.model.slice(0, maxCh - 1) + '…' : r.model;
      kids.push(svg('g', { class: 'row', tabindex: '0' },
        svg('title', {}, `${r.model}\nIn-Domain ${f3(r.in_domain)} → Out-of-Domain ${f3(r.out_of_domain)}`),
        svg('rect', { x: 0, y: y - rowH / 2, width: W, height: rowH, fill: 'transparent' }),
        svg('text', { class: 'lbl', x: L - 10, y: y + 4, 'text-anchor': 'end' }, label),
        svg('line', { class: 'stem', x1: x(r.in_domain), x2: x(r.out_of_domain), y1: y, y2: y, stroke: c, 'stroke-width': 2, 'stroke-opacity': .7, 'stroke-linecap': 'round' }),
        svg('circle', { cx: x(r.in_domain), cy: y, r: 4.5, fill: '#000', stroke: c, 'stroke-width': 2 }),
        svg('circle', { cx: x(r.out_of_domain), cy: y, r: 5, fill: c, stroke: '#000', 'stroke-width': 2 })));
    });
    node.replaceChildren(node.firstElementChild, ...kids);
  };
  draw();
  let t; addEventListener('resize', () => { clearTimeout(t); t = setTimeout(draw, 120); });
}

/* ------------------------------------------------- categories & efficiency */
function renderSmallTable(id, rows, cols, defaultShow = 6) {
  const table = document.getElementById(id), btn = $(`.showall[data-for="${id}"]`);
  table.tHead.replaceChildren(el('tr', {}, ...cols.map(c => el('th', { class: c.n ? 'n' : '', scope: 'col' }, c.label))));
  table.tBodies[0].replaceChildren(...rows.map((r, i) => el('tr', { hidden: i >= defaultShow },
    ...cols.map(c => el('td', { class: c.n ? 'n' : '' }, c.fmt ? c.fmt(r[c.k], r) : (r[c.k] ?? '—'))))));
  let open = false;
  if (rows.length <= defaultShow) btn.hidden = true;
  btn.addEventListener('click', () => { open = !open; [...table.tBodies[0].rows].forEach((tr, i) => tr.hidden = !open && i >= defaultShow); btn.textContent = open ? 'Show fewer' : 'Show all agents'; });
}
function renderCategories(cats) {
  if (!cats?.length) { status($('#cats tbody'), 'No category table.'); return; }
  const keys = Object.keys(cats[0]).filter(k => k !== 'model');
  const rows = [...cats].sort((a, b) => (b.overall ?? 0) - (a.overall ?? 0));
  renderSmallTable('cats', rows, [{ k: 'model', label: 'Model' }, ...keys.map(k => ({ k, label: k === 'overall' ? 'Overall' : k.slice(0, 5), n: true, fmt: f3 }))]);
}
function renderEfficiency(eff) {
  if (!eff?.length) { status($('#eff tbody'), 'No efficiency table.'); return; }
  const rows = [...eff].sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
  renderSmallTable('eff', rows, [
    { k: 'model', label: 'Model' }, { k: 'score', label: 'Score', n: true, fmt: f3 },
    { k: 'produced', label: 'Videos', n: true, fmt: (v, r) => `${fint(v)}${r.attempts ? '/' + fint(r.attempts) : ''}` },
    { k: 'tool_use_rate', label: 'Tool use', n: true, fmt: f2 }, { k: 'tool_calls_per_attempt', label: 'Calls', n: true, fmt: v => v == null ? '—' : (+v).toFixed(1) },
    { k: 'timeouts', label: 'Timeouts', n: true, fmt: fint }, { k: 'median_seconds', label: 'Median s', n: true, fmt: fint },
    { k: 'mean_input_tokens', label: 'In tok', n: true, fmt: fint }, { k: 'mean_output_tokens', label: 'Out tok', n: true, fmt: fint }]);
}

/* ---------------------------------------------------------------- heatmap */
const tip = $('#tip');
function showTip(e, html) { tip.innerHTML = html; tip.style.display = 'block'; moveTip(e); }
function moveTip(e) { const w = tip.offsetWidth, h = tip.offsetHeight; let x = e.clientX + 14, y = e.clientY + 14; if (x + w > innerWidth - 8) x = e.clientX - w - 10; if (y + h > innerHeight - 8) y = e.clientY - h - 10; tip.style.left = x + 'px'; tip.style.top = y + 'px'; }
function hideTip() { tip.style.display = 'none'; }
const heatColor = v => `hsl(212 ${Math.round(60 + 40 * v)}% ${Math.round(9 + 49 * Math.max(0, Math.min(1, v)))}%)`;

/* one column per task (tasks[].mean, or the mean of instance scores), one row per closed lane */
function renderHeatmap(tasks, lanes, laneList, gotoTask) {
  const node = $('#hm');
  const agents = laneList.length ? laneList : [...new Set(tasks.flatMap(t => Object.keys(t.mean || t.instances?.[0]?.lanes || {})))];
  const meanOf = (t, l) => { if (t.mean && t.mean[l] != null) return num(t.mean[l]); const s = (t.instances || []).map(i => num(i.lanes?.[l]?.score)).filter(v => v != null); return s.length ? s.reduce((a, b) => a + b, 0) / s.length : null; };
  const rows = tasks.map(t => ({ id: t.id, split: splitOf(t.split), category: t.category || '', scores: agents.map(l => meanOf(t, l)) }));
  const splits = ['In-Domain', 'Out-of-Domain'];
  const ordered = [];
  for (const s of splits) ordered.push(...rows.filter(r => r.split === s).sort((a, b) => a.category.localeCompare(b.category) || a.id.localeCompare(b.id)));
  ordered.push(...rows.filter(r => !splits.includes(r.split)));
  const L = 74, T = 46, cw = 11, ch = 26, gap = 12, W = L + ordered.length * cw + gap + 8, H = T + agents.length * ch + 8;
  node.setAttribute('viewBox', `0 0 ${W} ${H}`); node.setAttribute('width', W); node.setAttribute('height', H); node.style.minWidth = W + 'px';
  const kids = [];
  const xOf = i => L + i * cw + (ordered[i].split === 'Out-of-Domain' ? gap : 0);
  const shortLane = l => (lanes[l]?.label || l).split('/')[0].replace(' CLI', '').replace(' Code', '').trim();
  agents.forEach((a, j) => kids.push(svg('text', { x: L - 8, y: T + j * ch + ch / 2 + 4, 'text-anchor': 'end' }, shortLane(a))));
  let i = 0;
  while (i < ordered.length) {
    let j = i; while (j < ordered.length && ordered[j].split === ordered[i].split && ordered[j].category === ordered[i].category) j++;
    const x0 = xOf(i), x1 = xOf(j - 1) + cw, lbl = ordered[i].category || '';
    kids.push(svg('line', { class: 'band', x1: x0 + 1, x2: x1 - 1, y1: T - 6, y2: T - 6 }));
    const short = (x1 - x0) < lbl.length * 6.2 ? lbl.slice(0, Math.max(1, Math.floor((x1 - x0) / 6.2) - 1)) + (lbl.length > 3 ? '.' : '') : lbl;
    if (lbl) kids.push(svg('text', { class: 'bandl', x: (x0 + x1) / 2, y: T - 11, 'text-anchor': 'middle' }, svg('title', {}, lbl), short));
    i = j;
  }
  for (const s of splits) {
    const idx = ordered.map((r, k) => r.split === s ? k : -1).filter(k => k >= 0);
    if (!idx.length) continue;
    kids.push(svg('text', { class: 'splitl', x: (xOf(idx[0]) + xOf(idx[idx.length - 1]) + cw) / 2, y: 14, 'text-anchor': 'middle' }, `${s} (${idx.length})`));
  }
  ordered.forEach((r, k) => agents.forEach((a, j) => {
    const v = r.scores[j], name = lanes[a]?.label || a;
    const rect = svg('rect', { class: 'cell', x: xOf(k), y: T + j * ch, width: cw, height: ch, rx: 2, fill: v == null ? '#1a1a1a' : heatColor(v), tabindex: '0', role: 'button', 'aria-label': `${r.id}, ${name}: ${f3(v)}` });
    const html = `<b>${esc(r.id)}</b><br><span class="d">${esc(r.split)} · ${esc(r.category)}</span><br>${esc(name)}: <b>${f3(v)}</b><br><span class="d">click to open the task</span>`;
    rect.addEventListener('mouseenter', e => showTip(e, html)); rect.addEventListener('mousemove', moveTip); rect.addEventListener('mouseleave', hideTip);
    rect.addEventListener('click', () => { hideTip(); gotoTask(r.id); });
    rect.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); gotoTask(r.id); } });
    kids.push(rect);
  }));
  node.replaceChildren(...kids);
}

/* ---------------------------------------------------------------- gallery */
let LANES = {};
const laneName = l => LANES[l]?.label?.replace(/ \(OpenCode\)$/, '').split(' / ')[0] || l.replace(/^bedrock-/, '');
const laneFull = l => LANES[l]?.label || l;
const laneKind = l => LANES[l] ? kindOf(LANES[l]) : (['codex', 'claude', 'gemini'].includes(l) ? 'agent' : 'open');

function videoFrame({ video, poster, tag, score, kind, label }) {
  const frame = el('div', { class: 'frame', style: kind ? `--k:var(--${kind})` : '' });
  const v = el('video', { muted: true, loop: true, playsinline: true, preload: 'none', poster: poster || null, 'aria-label': label || tag, tabindex: '0' });
  v.muted = true; // attribute + property, for autoplay policies
  if (video) v.src = video; else frame.classList.add('missing');
  v.addEventListener('error', () => { frame.classList.add('missing'); frame.classList.remove('playing'); }, { once: true });
  const play = () => { if (!video || frame.classList.contains('missing')) return; const p = v.play(); if (p && p.catch) p.catch(() => {}); frame.classList.add('playing'); };
  const pause = () => { v.pause(); frame.classList.remove('playing'); };
  frame.addEventListener('mouseenter', play); frame.addEventListener('mouseleave', pause);
  v.addEventListener('click', () => frame.classList.contains('playing') ? pause() : play());
  v.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); frame.classList.contains('playing') ? pause() : play(); } });
  frame.append(v);
  if (tag) frame.append(el('span', { class: 'tag' }, tag));
  if (score != null) frame.append(el('span', { class: 'badge' }, f3(score)));
  frame.append(el('span', { class: 'play', 'aria-hidden': 'true' }, el('i')), el('span', { class: 'nov' }, 'no video'));
  return frame;
}

const codeCache = new Map();
async function fetchText(path) {
  if (!codeCache.has(path)) codeCache.set(path, fetch(path).then(r => r.ok ? r.text() : Promise.reject(new Error(`HTTP ${r.status}`))));
  return codeCache.get(path);
}

function taskCard(t, closedLanes) {
  const lanes = closedLanes.length ? closedLanes : Object.keys(t.instances?.[0]?.lanes || {});
  const card = el('article', { class: 'task', id: 'task-' + t.id, dataset: { id: t.id } });
  const split = splitOf(t.split), isID = split === 'In-Domain';
  const instBtns = (t.instances || []).map((ins, k) => el('button', { class: 'ibtn' + (k === 0 ? ' on' : ''), type: 'button', 'aria-label': `instance ${idxNum(ins.idx ?? k)}`, 'aria-pressed': k === 0 ? 'true' : 'false', onclick: () => setInstance(k) }, idxNum(ins.idx ?? k)));
  const head = el('div', { class: 'thead' },
    el('span', { class: 'id' }, t.id),
    t.short && t.short !== t.id ? el('span', { class: 'short' }, t.short.replace(/^[GO]-\d+_/, '').replace(/_/g, ' ')) : null,
    el('span', { class: 'pill split-' + (isID ? 'id' : 'ood') }, split),
    t.category ? el('span', { class: 'pill' }, t.category) : null,
    instBtns.length > 1 ? el('div', { class: 'inst', role: 'group', 'aria-label': 'instance' }, el('span', {}, 'instance'), ...instBtns) : null);
  const body = el('div', { class: 'tbody lazy' });
  card.append(head, body);
  let cur = 0, built = false, codeBox = null, codeLane = null, openBox = null;

  const build = () => {
    const ins = t.instances?.[cur]; if (!ins) { body.replaceChildren(el('div', { class: 'status' }, 'no instances')); return; }
    const gt = ins.gt || t.gt || {}, promptText = ins.prompt || t.prompt || '', label = idxNum(ins.idx ?? cur);
    const first = el('div', { class: 'frame' }, el('img', { src: gt.first_frame || '', alt: `first frame of ${t.id} instance ${label}`, loading: 'lazy', decoding: 'async' }), el('span', { class: 'tag' }, 'first frame'));
    const prompt = el('div', { class: 'prompt' }, promptText);
    const more = el('button', { class: 'more', type: 'button', onclick: () => { prompt.classList.toggle('open'); more.textContent = prompt.classList.contains('open') ? 'Less' : 'Full prompt'; } }, 'Full prompt');
    const left = el('div', { class: 'tprompt' }, first, prompt, promptText.length > 420 ? more : null);
    const vids = el('div', { class: 'vids' });
    vids.append(el('div', { class: 'lane k-gt' }, el('div', { class: 'ln' }, el('b', {}, el('i'), 'Ground truth')),
      videoFrame({ video: gt.video, poster: gt.last_frame, tag: 'GT', label: `ground truth ${t.id} ${label}` })));
    for (const l of lanes) {
      const d = ins.lanes?.[l] || {}, kind = laneKind(l), solve = d.solve_url || d.solve;
      const btn = solve ? el('button', { class: 'cbtn', type: 'button', 'aria-expanded': 'false', onclick: () => toggleCode(l, solve, btn) }, 'program') : null;
      vids.append(el('div', { class: 'lane k-' + kind }, el('div', { class: 'ln' }, el('b', { title: laneFull(l) }, el('i'), laneName(l)), btn),
        videoFrame({ video: d.video, poster: d.last_frame, tag: laneName(l), score: d.score, kind, label: `${laneFull(l)} ${t.id} ${label}` })));
    }
    body.replaceChildren(left, vids);
    const ex = (t.open_examples || []).filter(o => o.video);
    if (ex.length) {
      const obtn = el('button', { class: 'cbtn obtn', type: 'button', 'aria-expanded': 'false', onclick: () => toggleOpen(ex, obtn) }, `open-weight attempts (${ex.length})`);
      body.append(el('div', { class: 'orow' }, obtn));
    }
    body.classList.remove('lazy');
    codeBox = null; codeLane = null; openBox = null; built = true;
  };
  const toggleOpen = (ex, btn) => {
    if (openBox) { openBox.remove(); openBox = null; btn.classList.remove('on'); btn.setAttribute('aria-expanded', 'false'); return; }
    openBox = el('div', { class: 'vids open-vids' }, ...ex.map(o => el('div', { class: 'lane k-open' },
      el('div', { class: 'ln' }, el('b', { title: laneFull(o.lane) }, el('i'), laneName(o.lane)), el('span', { class: 'idx' }, `instance ${idxNum(o.idx)}`)),
      videoFrame({ video: o.video, poster: o.last_frame, tag: laneName(o.lane), score: o.score, kind: 'open', label: `${laneFull(o.lane)} ${t.id} ${idxNum(o.idx)}` }))));
    btn.parentNode.after(openBox); btn.classList.add('on'); btn.setAttribute('aria-expanded', 'true');
  };
  const toggleCode = async (lane, path, btn) => {
    body.querySelectorAll('.cbtn:not(.obtn)').forEach(b => { b.classList.remove('on'); b.setAttribute('aria-expanded', 'false'); });
    if (codeBox && codeLane === lane) { codeBox.remove(); codeBox = null; codeLane = null; return; }
    if (codeBox) codeBox.remove();
    const pre = el('pre', {}, 'Loading…');
    codeBox = el('div', { class: 'code' }, el('div', { class: 'ch' }, el('span', {}, `${laneName(lane)} · ${path.split('/').slice(-3).join('/')}`), el('a', { href: path, target: '_blank', rel: 'noopener' }, 'raw')), pre);
    codeLane = lane; body.append(codeBox); btn.classList.add('on'); btn.setAttribute('aria-expanded', 'true');
    try { pre.textContent = await fetchText(path); } catch (e) { pre.replaceChildren(el('span', { class: 'err' }, `could not load ${path} (${e.message})`)); }
  };
  const setInstance = k => { cur = k; instBtns.forEach((b, i) => { b.classList.toggle('on', i === k); b.setAttribute('aria-pressed', i === k ? 'true' : 'false'); }); if (built || card._visible) build(); };
  card._build = () => { if (!built) build(); };
  return card;
}

function renderGallery(tasks, closedLanes) {
  const box = $('#tasks'), fSplit = $('#f-split'), fCat = $('#f-cat'), fQ = $('#f-q'), cnt = $('#f-cnt');
  const cats = [...new Set(tasks.map(t => t.category).filter(Boolean))].sort();
  fCat.append(...cats.map(c => el('option', { value: c }, c)));
  const cards = new Map(tasks.map(t => [t.id, taskCard(t, closedLanes)]));
  const io = new IntersectionObserver(entries => { for (const e of entries) if (e.isIntersecting) { e.target._visible = true; e.target._build(); io.unobserve(e.target); } }, { rootMargin: '400px 0px' });
  const hay = new Map(tasks.map(t => [t.id, [t.id, t.short, t.category, splitOf(t.split), t.prompt, ...(t.instances || []).map(i => i.prompt)].join(' ').toLowerCase()]));
  const apply = () => {
    const s = fSplit.value, c = fCat.value, q = fQ.value.trim().toLowerCase();
    const shown = tasks.filter(t => (!s || splitOf(t.split) === s) && (!c || t.category === c) && (!q || hay.get(t.id).includes(q)));
    box.replaceChildren(...shown.map(t => cards.get(t.id)));
    shown.forEach(t => { const cd = cards.get(t.id); if (!cd._visible) io.observe(cd); });
    cnt.textContent = `${shown.length} / ${tasks.length} tasks`;
    if (!shown.length) box.append(el('div', { class: 'status' }, 'No task matches.'));
  };
  [fSplit, fCat].forEach(x => x.addEventListener('change', apply));
  let t; fQ.addEventListener('input', () => { clearTimeout(t); t = setTimeout(apply, 120); });
  apply();
  const byNorm = new Map(tasks.map(t => [normId(t.id), t.id]));
  return id => {
    id = cards.has(id) ? id : byNorm.get(normId(id));
    if (!cards.has(id)) return;
    if (!cards.get(id).isConnected) { fSplit.value = ''; fCat.value = ''; fQ.value = ''; apply(); }
    const cd = cards.get(id); cd._visible = true; cd._build();
    cd.scrollIntoView({ behavior: 'smooth', block: 'start' });
    cd.classList.add('flash'); setTimeout(() => cd.classList.remove('flash'), 2200);
  };
}

/* ------------------------------------------------------- open examples */
/* a sample: the best-scoring attempt per open lane per task, round-robin over lanes, one card per task, 12 cards */
function renderOpen(tasks, gotoTask) {
  const grid = $('#ogrid');
  const all = tasks.flatMap(t => (t.open_examples || []).filter(o => o.video).map(o => ({ ...o, task: t }))).sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
  const byLane = new Map(); for (const o of all) { if (!byLane.has(o.lane)) byLane.set(o.lane, []); byLane.get(o.lane).push(o); }
  const pick = [], seen = new Set(), lanes = [...byLane.keys()];
  for (let round = 0; pick.length < 12 && round < 40; round++) for (const l of lanes) {
    const o = (byLane.get(l) || []).find(o => !seen.has(o.task.id) && !seen.has('o:' + o.lane + o.task.id + o.idx));
    if (o && pick.length < 12) { pick.push(o); seen.add(o.task.id); seen.add('o:' + o.lane + o.task.id + o.idx); }
  }
  if (!pick.length) { grid.closest('section').hidden = true; return; }
  grid.replaceChildren(...pick.map(o => el('div', { class: 'ocard' },
    videoFrame({ video: o.video, poster: o.last_frame, tag: laneName(o.lane), score: o.score, kind: 'open', label: `${laneFull(o.lane)} ${o.task.id} ${idxNum(o.idx)}` }),
    el('div', { class: 'om' }, el('b', { title: laneFull(o.lane) }, laneName(o.lane)),
      el('a', { href: '#task-' + o.task.id, onclick: e => { e.preventDefault(); gotoTask(o.task.id); } }, `${o.task.id.split('_')[0]} · ${idxNum(o.idx)}`)))));
  const n = all.length, nl = lanes.length;
  const sub = $('#open .sub'); if (sub && n) sub.textContent = `${n} videos rendered by programs from ${nl} open-weight models (OpenCode on Amazon Bedrock), two per model per task; a sample of the best below, the rest under each task card. Each card names the model, the task and its official score.`;
}

/* ------------------------------------------------------------------ boot */
$('#copybib').addEventListener('click', async e => {
  const txt = $('#bib').textContent.replace(/^Copy/, '').trim();
  try { await navigator.clipboard.writeText(txt); e.target.textContent = 'Copied'; } catch { e.target.textContent = 'Select & copy'; }
  setTimeout(() => e.target.textContent = 'Copy', 1600);
});

(async () => {
  let lbFile;
  try { lbFile = await loadJSON('data/leaderboard.json'); } catch (err) { status($('#lb tbody'), `Could not load data/leaderboard.json (${err.message})`, true); return; }
  const rowsRaw = Array.isArray(lbFile) ? lbFile : (lbFile.rows || []);
  const cats = Array.isArray(lbFile) ? null : lbFile.categories, eff = Array.isArray(lbFile) ? null : lbFile.efficiency;
  const catsP = cats ? Promise.resolve(cats) : loadJSON('data/categories.json').catch(() => []);
  const effP = eff ? Promise.resolve(eff) : loadJSON('data/efficiency.json').catch(() => []);
  const lb = prepareLeaderboard(rowsRaw, await effP);
  renderHero(lb); renderLeaderboard(lb); renderDumbbell(lb);
  catsP.then(renderCategories); effP.then(renderEfficiency);

  let tf;
  try { tf = await loadJSON('data/tasks.json'); } catch (err) {
    status($('#tasks'), `Could not load data/tasks.json (${err.message})`, true); status($('#ogrid'), 'No open-weight examples.');
    $('#hm').replaceWith(el('div', { class: 'status err' }, 'Per-task scores need data/tasks.json.')); return;
  }
  const tasks = Array.isArray(tf) ? tf : (tf.tasks || []);
  LANES = (tf.lanes && !Array.isArray(tf)) ? tf.lanes : {};
  const closed = (!Array.isArray(tf) && tf.closed_lanes) || Object.keys(tasks[0]?.instances?.[0]?.lanes || {});
  let gotoTask = id => { location.hash = 'task-' + id; };
  if (tasks.length) { gotoTask = renderGallery(tasks, closed); renderOpen(tasks, gotoTask); renderHeatmap(tasks, LANES, closed, gotoTask); }
  else { status($('#tasks'), 'data/tasks.json has no tasks.', true); status($('#ogrid'), 'No open-weight examples.'); }
  if (location.hash.startsWith('#task-')) gotoTask(decodeURIComponent(location.hash.slice(6)));
})();
})();
