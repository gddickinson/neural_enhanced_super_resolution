// Compare: before/after slider on the precomputed examples (synthetic patterns, bicubic-downsampled by 2 or 4).
// Nearest, bicubic and Lanczos are recomputed here in the browser from the low-res PNG (resample.js) and checked
// against the Python/OpenCV output; model and pipeline outputs are the PNGs written by scripts/make_examples.py.
import { app, esc, list, route } from './core.js';
import { compare, fmtPsnr, fmtSsim, loadUrl, upscale } from './images.js';
import { psnr, ssim } from './metrics.js';
import { beforeAfter } from './slider.js';
import { exampleUrl, getExamples } from './state.js';

const BROWSER = new Set(['nearest', 'bicubic', 'lanczos']);
const imgCache = new Map();
const load = file => { if (!imgCache.has(file)) imgCache.set(file, loadUrl(exampleUrl(file))); return imgCache.get(file); };

const state = { pattern: 'siemens_star', factor: 2, left: 'bicubic', right: 'realesrgan', zoom: 1, pos: 50 };

route('compare', async (args, query) => {
  const m = await getExamples();
  if (!m.examples.length) { app().innerHTML = '<h1>Compare</h1><p class="err">No examples have been generated (scripts/make_examples.py).</p>'; return; }
  Object.assign(state, Object.fromEntries(Object.entries(query).filter(([k]) => k in state).map(([k, v]) => [k, k === 'factor' || k === 'zoom' ? +v : v])));
  const opts = [{ id: 'lr', label: 'Low-res input (each pixel as a block)' }, ...m.methods, { id: 'hr', label: 'Ground truth (original pattern)' }];
  const seg = (name, items) => `<div class="seg" role="group" aria-label="${name}">${list(items, ([v, t]) =>
    `<button type="button" data-k="${name}" data-v="${v}">${esc(t)}</button>`)}</div>`;
  const sel = (id, label) => `<div><label for="sel-${id}">${label}</label><select id="sel-${id}">${list(opts, o => `<option value="${o.id}">${esc(o.label)}</option>`)}</select></div>`;
  app().innerHTML = `<p class="kicker">Before / after</p><h1>Compare upscalers</h1>
    <p class="lead">Each ${m.size} × ${m.size} test pattern was shrunk by 2 or 4 with the repo's degradation tool
      (<code>downsample_image</code>, bicubic) and enlarged back by each method. Drag the divider to compare any two;
      the ground truth is the original pattern.</p>
    <div class="ctlrow">
      <div><label>Pattern</label>${seg('pattern', m.patterns.map(p => [p.id, p.title]))}</div>
      <div><label>Factor</label>${seg('factor', m.factors.map(f => [f, '×' + f]))}</div>
      <div><label>Zoom</label>${seg('zoom', [[1, '1×'], [2, '2×'], [4, '4×']])}</div>
    </div>
    <div class="ctlrow">${sel('left', 'Left image')}${sel('right', 'Right image (right of the divider)')}</div>
    <div id="ba"></div>
    <p class="small mute" id="note"></p>
    <h2>Scores against the ground truth</h2>
    <p class="small mute">PSNR in dB and SSIM (1 = identical). <span class="badge py">Python</span> from examples.json
      (nesr/metrics.py); <span class="badge js">browser</span> recomputed now by metrics.js from the pixels shown. Higher is better.</p>
    <div class="scroll"><table id="scores"></table></div>`;
  for (const k of ['left', 'right']) {
    const s = document.getElementById('sel-' + k);
    s.value = state[k];
    s.addEventListener('change', () => { state[k] = s.value; draw(m); });
  }
  app().querySelectorAll('.seg button').forEach(b => b.addEventListener('click', () => {
    state[b.dataset.k] = b.dataset.k === 'pattern' ? b.dataset.v : +b.dataset.v; draw(m, b.dataset.k !== 'zoom');
  }));
  await draw(m, true);
});

async function images(ex) {
  const [hr, lr] = await Promise.all([load(ex.hr), load(ex.lr)]);
  const out = { hr, lr: upscale(lr, 'nearest', hr.w, hr.h) };
  const py = {};
  for (const [id, o] of Object.entries(ex.outputs)) {
    py[id] = await load(o.file);
    out[id] = BROWSER.has(id) ? upscale(lr, id, hr.w, hr.h) : py[id];
  }
  return { out, py };
}

let token = 0;
async function draw(m, scores = false) {
  const my = ++token;
  app().querySelectorAll('.seg button').forEach(b => b.classList.toggle('on', String(state[b.dataset.k]) === b.dataset.v));
  const ex = m.examples.find(e => e.pattern === state.pattern && e.factor === state.factor);
  const { out, py } = await images(ex);
  if (my !== token) return;
  const name = id => id === 'lr' ? `Low-res ${ex.lr_size}px` : id === 'hr' ? 'Ground truth' : m.methods.find(x => x.id === id).label;
  const host = document.getElementById('ba');
  const prev = host.querySelector('#ba-pos');
  beforeAfter(host, { left: out[state.left], right: out[state.right], leftLabel: name(state.left),
    rightLabel: name(state.right), zoom: state.zoom, pos: prev ? +prev.value : state.pos });
  const desc = id => m.methods.find(x => x.id === id)?.description || '';
  document.getElementById('note').innerHTML = [state.left, state.right].filter(id => desc(id))
    .map(id => `<b>${esc(name(id))}</b>: ${esc(desc(id))}`).join('<br>');
  if (scores) table(m, ex, out, py);
}

function table(m, ex, out, py) {
  const hr = out.hr, rows = m.methods.filter(x => ex.outputs[x.id]);
  const res = rows.map(x => {
    const o = ex.outputs[x.id], im = out[x.id];
    const js = { psnr: psnr(hr.data, im.data, hr.w, hr.h, 4), ssim: ssim(hr.data, im.data, hr.w, hr.h, 4) };
    const check = BROWSER.has(x.id) ? compare(im, py[x.id]) : null;
    return { x, o, js, check };
  });
  const best = k => Math.max(...res.map(r => r.o[k]));
  document.getElementById('scores').innerHTML = `<thead><tr><th>Method</th><th class="num">PSNR <span class="badge py">Python</span></th>
    <th class="num">PSNR <span class="badge js">browser</span></th><th class="num">SSIM <span class="badge py">Python</span></th>
    <th class="num">SSIM <span class="badge js">browser</span></th><th>Where it ran</th></tr></thead><tbody>${list(res, r => `<tr>
    <td>${esc(r.x.label)}</td><td class="num ${r.o.psnr === best('psnr') ? 'best' : ''}">${fmtPsnr(r.o.psnr)}</td><td class="num">${fmtPsnr(r.js.psnr)}</td>
    <td class="num ${r.o.ssim === best('ssim') ? 'best' : ''}">${fmtSsim(r.o.ssim)}</td><td class="num">${fmtSsim(r.js.ssim)}</td>
    <td class="small">${r.check ? `<span class="badge js">your browser</span> ${r.check.same === 1 ? 'identical to OpenCV' : `${(100 * r.check.same).toFixed(2)}% identical to OpenCV (max diff ${r.check.maxDiff})`}`
      : `<span class="badge py">precomputed</span> ${r.o.models_loaded ? 'models: ' + esc(r.o.models_loaded.join(', ') || 'none') : ''}`}</td></tr>`)}</tbody>`;
}
