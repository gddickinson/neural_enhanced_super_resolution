// Upscale: run one method on your own image or a test pattern. "Upscale" enlarges the image as it is;
// "Benchmark" first shrinks it (bicubic, like the GUI's degradation tool), restores it, and scores the result.
// Local app: every method runs in Python. Static site: nearest / bicubic / Lanczos run in the browser.
import { api, app, esc, isStatic, list, route } from './core.js';
import { fmtPsnr, fmtSsim, loadUrl, upscale } from './images.js';
import { beforeAfter } from './slider.js';
import { getInfo, runnable } from './state.js';

route('upscale', async () => {
  const info = await getInfo();
  const methods = runnable(info, isStatic());
  app().innerHTML = `<p class="kicker">Your image</p><h1>Upscale</h1>
    <p class="lead">Pick an image and a method. <b>Upscale</b> enlarges it as it is; <b>Benchmark</b> first shrinks it by the
      same factor (bicubic), restores it and reports PSNR and SSIM against your original.</p>
    ${isStatic() ? `<div class="callout">Static site: nearest, bicubic and Lanczos run here in your browser (an exact port of
      OpenCV's resize); the image never leaves your computer. The NESR pipeline and Real-ESRGAN need the local app (see About).</div>` : ''}
    <form class="panel" id="f">
      <div class="two">
        <div>
          <label for="file">Image file (PNG, JPEG, ...)</label><input id="file" name="file" type="file" accept="image/*">
          <label for="pattern">or a test pattern</label>
          <select id="pattern" name="pattern">${list(info.patterns, p => `<option value="${p.id}">${esc(p.title)}</option>`)}</select>
          ${isStatic() ? '' : `<label for="size">Pattern size (px)</label><input id="size" name="size" type="number" min="16" max="1024" value="256">`}
        </div>
        <div>
          <label for="method">Method</label>
          <select id="method" name="method">${list(methods, m => `<option value="${m.id}" ${m.ok ? '' : 'disabled'}>${esc(m.label)}${m.ok ? '' : ' (' + esc(m.why) + ')'}</option>`)}</select>
          <p class="small mute" id="mdesc"></p>
          <div class="row">
            <div><label for="scale">Factor</label><select id="scale" name="scale">${list(info.scales, s => `<option value="${s}">×${s}</option>`)}</select></div>
            <div><label for="mode">Mode</label><select id="mode" name="mode"><option value="upscale">Upscale</option><option value="benchmark" selected>Benchmark (shrink, restore, score)</option></select></div>
          </div>
        </div>
      </div>
      <button type="submit">Run</button><span class="small mute" id="status" role="status"></span>
    </form>
    <div id="out"></div>`;
  const f = document.getElementById('f'), sel = document.getElementById('method');
  const showDesc = () => { document.getElementById('mdesc').textContent = methods.find(m => m.id === sel.value)?.description || ''; };
  sel.value = 'bicubic'; showDesc(); sel.addEventListener('change', showDesc);
  f.addEventListener('submit', async e => {
    e.preventDefault();
    const btn = f.querySelector('button'), status = document.getElementById('status');
    btn.disabled = true; status.textContent = ' running...';
    try {
      const r = await api('/upscale', { body: new FormData(f) });
      status.textContent = ` done in ${r.info.seconds ?? '?'} s`;
      await show(r);
    } catch (err) {
      status.textContent = '';
      document.getElementById('out').innerHTML = `<p class="err">${esc(err.message)}</p>`;
    } finally { btn.disabled = false; }
  });
});

async function show(r) {
  const out = await loadUrl(r.output);
  const inp = await loadUrl(r.input);
  const ref = r.reference ? await loadUrl(r.reference) : null;
  const left = ref && ref.w === out.w && ref.h === out.h ? ref : upscale(inp, 'nearest', out.w, out.h);
  const host = document.getElementById('out');
  const m = r.metrics;
  host.innerHTML = `<h2>Result</h2>
    <p class="small mute">${esc(r.source)} · input ${r.input_size.join(' × ')} → output ${r.output_size.join(' × ')} px ·
      ${esc(r.info.label)}${r.info.device ? ' on ' + esc(r.info.device) : ''}${r.info.models_loaded ? ' · models loaded: ' + esc(r.info.models_loaded.join(', ') || 'none (bicubic fallback)') : ''}
      ${r.info.where ? ' · ' + esc(r.info.where) : ''}</p>
    ${m ? `<div class="tiles"><div class="tile"><b>${fmtPsnr(m.psnr)}</b><span>PSNR (dB) vs your original</span></div>
      <div class="tile"><b>${fmtSsim(m.ssim)}</b><span>SSIM vs your original</span></div></div>` : ''}
    ${r.metrics_note ? `<p class="small mute">${esc(r.metrics_note)}</p>` : ''}
    <div id="ba2"></div>
    <a class="btn alt" download="nesr_${esc(r.info.method)}_x${r.info.scale}.png" href="${r.output}">Download result PNG</a>`;
  beforeAfter(document.getElementById('ba2'), { left, right: out,
    leftLabel: left === ref ? 'Original' : 'Input (pixels as blocks)', rightLabel: r.info.label });
}
