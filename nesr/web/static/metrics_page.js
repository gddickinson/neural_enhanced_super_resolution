// Metrics: the benchmark table of every precomputed example (Python PSNR / SSIM), per-pixel SSIM and error maps
// computed in the browser, the formulas, and a one-click check that metrics.js reproduces the Python numbers.
import { app, esc, list, route } from './core.js';
import { fmtPsnr, fmtSsim, loadUrl, toCanvas } from './images.js';
import { gaussianKernel, psnr, ssim, ssimMap } from './metrics.js';
import { exampleUrl, getExamples } from './state.js';

route('metrics', async () => {
  const m = await getExamples();
  const meth = m.methods;
  const wins = Object.fromEntries(meth.map(x => [x.id, { psnr: 0, ssim: 0 }]));
  for (const e of m.examples) for (const k of ['psnr', 'ssim']) {
    const best = Math.max(...Object.values(e.outputs).map(o => o[k]));
    for (const [id, o] of Object.entries(e.outputs)) if (o[k] === best) wins[id][k]++;
  }
  const k = gaussianKernel();
  app().innerHTML = `<p class="kicker">Quality metrics</p><h1>PSNR and SSIM</h1>
    <p class="lead">How close each method gets to the original pattern, for all ${m.examples.length} examples
      (${m.patterns.length} patterns × factors ${m.factors.join(' and ')}). Numbers from <code>nesr/metrics.py</code>; best per row in colour.</p>
    <div class="ctlrow"><div><label for="metric">Metric</label><select id="metric"><option value="psnr">PSNR (dB)</option><option value="ssim">SSIM</option></select></div></div>
    <div class="scroll"><table id="bench"></table></div>
    <p class="small mute">Generated on ${esc(m.versions?.machine || '?')} (CPU) with OpenCV ${esc(m.versions?.opencv || '?')}${m.versions?.torch ? `, torch ${esc(m.versions.torch)}, basicsr ${esc(m.versions.basicsr)}, realesrgan ${esc(m.versions.realesrgan)}` : ''}.
      Degradation: ${esc(m.degradation || '')}. Synthetic patterns are a hard, unusual test: Real-ESRGAN was trained on photographs with
      mixed blur, noise and JPEG degradations, and the pipeline's CLAHE contrast step changes tones that PSNR counts as error.</p>
    <h2>Error maps</h2>
    <div class="ctlrow">
      <div><label for="ex">Example</label><select id="ex">${list(m.examples, e => `<option value="${e.id}">${esc(e.title)} ×${e.factor}</option>`)}</select></div>
      <div><label for="mm">Method</label><select id="mm">${list(meth, x => `<option value="${x.id}">${esc(x.label)}</option>`)}</select></div>
    </div>
    <div class="maps" id="maps"></div>
    <h2>Definitions</h2>
    <p><b>PSNR</b> (peak signal-to-noise ratio) compares pixel values directly:</p>
    <p class="eq">PSNR = 10 · log10( 255² / MSE ),  MSE = mean over pixels and R, G, B of (x − y)²</p>
    <p><b>SSIM</b> (structural similarity, Wang et al. 2004) compares local means, contrasts and correlation:</p>
    <p class="eq">SSIM = (2μxμy + C1)(2σxy + C2) / ((μx² + μy² + C1)(σx² + σy² + C2))</p>
    <p class="small">Local statistics use an ${k.length} × ${k.length} Gaussian window (σ = 1.5), C1 = (0.01·255)², C2 = (0.03·255)²,
      mirrored borders; the map is averaged over the image minus a 5-pixel border and over the three colour channels. These are the
      settings of <code>skimage.metrics.structural_similarity(gaussian_weights=True, sigma=1.5, use_sample_covariance=False)</code>,
      which the Python tests check against.</p>
    <h2>Browser vs Python</h2>
    <p>The static site computes the metrics with <code>metrics.js</code>, a port of <code>nesr/metrics.py</code> (node tests compare them on
      fixed image pairs). Check it on every precomputed output:</p>
    <button type="button" id="verify">Recompute all ${m.examples.length * meth.length} scores here</button>
    <p id="vres" class="small" role="status"></p>`;
  const metric = document.getElementById('metric');
  const bench = () => {
    const key = metric.value, f = key === 'psnr' ? fmtPsnr : fmtSsim;
    document.getElementById('bench').innerHTML = `<thead><tr><th>Pattern</th><th>Factor</th>${list(meth, x => `<th class="num" title="${esc(x.label)}">${esc(x.short || x.label)}</th>`)}</tr></thead><tbody>${
      list(m.examples, e => { const best = Math.max(...Object.values(e.outputs).map(o => o[key]));
        return `<tr><td>${esc(e.title)}</td><td>×${e.factor}</td>${list(meth, x => { const o = e.outputs[x.id];
          return `<td class="num ${o && o[key] === best ? 'best' : ''}">${o ? f(o[key]) : '—'}</td>`; })}</tr>`; })}
      <tr><td colspan="2" class="mute">Best in how many rows</td>${list(meth, x => `<td class="num">${wins[x.id][key]}</td>`)}</tr></tbody>`;
  };
  metric.addEventListener('change', bench); bench();
  const exSel = document.getElementById('ex'), mmSel = document.getElementById('mm');
  exSel.value = 'siemens_star_x2'; mmSel.value = 'realesrgan';
  const maps = () => drawMaps(m.examples.find(e => e.id === exSel.value), mmSel.value, meth);
  exSel.addEventListener('change', maps); mmSel.addEventListener('change', maps); await maps();
  document.getElementById('verify').addEventListener('click', () => verify(m));
});

async function drawMaps(e, id, meth) {
  const host = document.getElementById('maps');
  const o = e.outputs[id];
  if (!o) { host.innerHTML = '<p class="mute">No output for this method.</p>'; return; }
  const [hr, out] = await Promise.all([loadUrl(exampleUrl(e.hr)), loadUrl(exampleUrl(o.file))]);
  const { w, h } = hr, n = w * h;
  const err = { w, h, data: new Uint8ClampedArray(n * 4) }, sm = { w, h, data: new Uint8ClampedArray(n * 4) };
  const acc = new Float64Array(n);
  for (let c = 0; c < 3; c++) {
    const a = new Float64Array(n), b = new Float64Array(n);
    for (let i = 0; i < n; i++) { a[i] = hr.data[i * 4 + c]; b[i] = out.data[i * 4 + c]; }
    const s = ssimMap(a, b, w, h);
    for (let i = 0; i < n; i++) acc[i] += s[i] / 3;
  }
  for (let i = 0; i < n; i++) {
    let d = 0;
    for (let c = 0; c < 3; c++) d += Math.abs(hr.data[i * 4 + c] - out.data[i * 4 + c]) / 3;
    const v = Math.min(255, d * 2);
    err.data.set([v, v * 0.6, Math.max(0, 60 - v / 4), 255], i * 4);
    const s = Math.max(0, Math.min(1, acc[i])) * 255;
    sm.data.set([s, s, s, 255], i * 4);
  }
  const label = meth.find(x => x.id === id).label;
  host.innerHTML = `<figure><div class="c1"></div><figcaption>Ground truth</figcaption></figure>
    <figure><div class="c2"></div><figcaption>${esc(label)}</figcaption></figure>
    <figure><div class="c3"></div><figcaption>|error| × 2 (bright = wrong)</figcaption></figure>
    <figure><div class="c4"></div><figcaption>SSIM map (white = 1, black ≤ 0)</figcaption></figure>`;
  [[hr, 'c1', 'Ground truth'], [out, 'c2', label], [err, 'c3', 'Absolute error map'], [sm, 'c4', 'SSIM map']].forEach(([im, cls, alt]) => {
    const c = toCanvas(im); c.setAttribute('role', 'img'); c.setAttribute('aria-label', alt); host.querySelector('.' + cls).appendChild(c);
  });
}

async function verify(m) {
  const res = document.getElementById('vres'), btn = document.getElementById('verify');
  btn.disabled = true;
  let dp = 0, ds = 0, n = 0;
  for (const e of m.examples) {
    const hr = await loadUrl(exampleUrl(e.hr));
    for (const o of Object.values(e.outputs)) {
      const out = await loadUrl(exampleUrl(o.file));
      dp = Math.max(dp, Math.abs(psnr(hr.data, out.data, hr.w, hr.h, 4) - o.psnr));
      ds = Math.max(ds, Math.abs(ssim(hr.data, out.data, hr.w, hr.h, 4) - o.ssim));
      res.textContent = `checked ${++n}...`;
      await new Promise(r => setTimeout(r, 0));
    }
  }
  res.innerHTML = `<span class="ok">${n} outputs checked.</span> Largest difference from Python: PSNR ${dp.toExponential(2)} dB,
    SSIM ${ds.toExponential(2)} (the stored values are rounded to 6 decimals).`;
  btn.disabled = false;
}
