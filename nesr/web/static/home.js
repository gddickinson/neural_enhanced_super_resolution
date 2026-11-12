// Home: hero, real counts, thumbnails of the test patterns and cards to the pages.
import { app, esc, isStatic, list, route } from './core.js';
import { exampleUrl, getExamples, getInfo } from './state.js';

route('home', async () => {
  const [info, m] = await Promise.all([getInfo(), getExamples()]);
  const nOut = m.examples.reduce((s, e) => s + Object.keys(e.outputs).length, 0);
  app().innerHTML = `
  <section class="hero">
    <p class="kicker">Iterative image super-resolution · multiple models</p>
    <h1 class="glow">Neural Enhanced Super-Resolution</h1>
    <p class="lead">NESR enlarges images in ×2 steps, combining <b>Real-ESRGAN</b>, a <b>Stable Diffusion x4 upscaler</b> and
      segmentation-guided sharpening with denoising, contrast enhancement and adaptive sharpening. This site compares it with the
      classical baselines, bicubic and Lanczos, and scores every result with PSNR and SSIM.</p>
    <a class="btn" href="#/compare">Compare upscalers</a><a class="btn alt" href="#/upscale">Upscale your image</a><a class="btn alt" href="#/pipeline">How the pipeline works</a>
  </section>
  <div class="tiles">
    <div class="tile"><b>${m.patterns.length}</b><span>synthetic test patterns</span></div>
    <div class="tile"><b>${m.examples.length}</b><span>examples (×${m.factors.join(', ×')})</span></div>
    <div class="tile"><b>${nOut}</b><span>precomputed outputs, each scored</span></div>
    <div class="tile"><b>${info.methods.length}</b><span>methods in the local app</span></div>
    <div class="tile"><b>${info.methods.filter(x => x.browser).length}</b><span>run in your browser</span></div>
  </div>
  ${isStatic() ? `<div class="callout">Static showcase: the model and pipeline outputs were computed by the repo's Python code
    (CPU, Real-ESRGAN x2plus) and saved; bicubic, Lanczos and the metrics are recomputed live in your browser by JavaScript ports
    tested against OpenCV and <code>nesr/metrics.py</code>. Running the models on your own images needs the local app (see About).</div>` : ''}
  <h2>Pages</h2>
  <div class="grid">
    <a class="card" href="#/compare"><b>Compare</b><span>Before/after slider on every example: any two methods, the low-res input or the ground truth, up to 4× zoom.</span></a>
    <a class="card" href="#/upscale"><b>Upscale</b><span>Your image or a pattern: enlarge it, or shrink-restore-score it as a benchmark.</span></a>
    <a class="card" href="#/metrics"><b>Metrics</b><span>PSNR and SSIM for all ${nOut} outputs, error and SSIM maps, and the formulas.</span></a>
    <a class="card" href="#/pipeline"><b>Pipeline</b><span>What each stage of <code>SuperResolutionPipeline</code> does, read from the code.</span></a>
  </div>
  <h2>Test patterns</h2>
  <p class="small mute">Generated from formulas by <code>nesr/test_patterns.py</code> (no third-party images).</p>
  <div class="thumbs">${list(m.patterns, p => `<a href="#/compare?pattern=${p.id}" class="card"><img src="${exampleUrl(p.id + '_hr.png')}"
    alt="${esc(p.title)} test pattern" width="256" height="256" loading="lazy"><b>${esc(p.title)}</b><span>${esc(p.description)}</span></a>`)}</div>`;
});
