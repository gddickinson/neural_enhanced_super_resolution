// Pipeline (what SuperResolutionPipeline does, read from nesr/nesr.py), the repo's markdown docs, and About.
import { api, app, esc, isStatic, list, route } from './core.js';
import { markdown, wireAnchors } from './md.js';
import { getInfo } from './state.js';

async function doc(name, kicker) {
  const d = await api('/docs/' + name);
  const src = d.markdown.replace(/!\[[^\]]*\]\([^)]*\)/g, '');
  const title = (src.match(/^#\s+(.*)$/m) || [, d.title])[1];
  app().innerHTML = `<p class="kicker">${esc(kicker)} · <span class="mono">${esc(d.file)}</span></p><h1>${esc(title)}</h1>
    <div class="md">${markdown(src, { skipTitle: true })}</div>`;
  wireAnchors(app());
}
route('readme', () => doc('readme', 'README'));
route('usage', () => doc('usage', 'Usage guide'));
route('structure', () => doc('structure', 'Project structure'));

route('pipeline', () => {
  app().innerHTML = `<p class="kicker">nesr/nesr.py · SuperResolutionPipeline.enhance_image</p><h1>How the pipeline works</h1>
  <p class="lead">Each iteration enlarges the image by <code>upscale_factor</code> (default 2; 3 iterations by default, so ×8 overall).
    Models load on first use; any that cannot load are skipped, and if none produce a result the step falls back to bicubic.</p>
  <ol class="stages">
    <li><b>Pre-processing</b> (<code>_preprocess_image</code>): OpenCV non-local-means colour denoising with strength
      <code>denoise_level × 10</code> (default 0.5 → 5), then CLAHE contrast enhancement (clip limit 2.0, 8 × 8 tiles) on the
      lightness channel in Lab colour space.</li>
    <li><b>Segmentation-guided sharpening</b> (<code>_segment_and_enhance</code>, if <code>segment_enhancement</code>): a SegFormer-B0
      (ADE20K) model labels regions; pixels outside class 0 get an unsharp mask (1.5 × image − 0.5 × Gaussian blur, σ = 3).</li>
    <li><b>Real-ESRGAN</b> (<code>_apply_esrgan</code>): RealESRGAN_x2plus (an RRDBNet GAN, 23 blocks), the weights committed in
      <code>models/weights/</code>. Large images are processed in tiles.</li>
    <li><b>Stable Diffusion x4 upscaler</b> (<code>_apply_diffusion</code>, if <code>use_diffusion</code>): text-guided diffusion
      (<code>stabilityai/stable-diffusion-x4-upscaler</code>, noise level 20, guidance 7.5, 10/15/20 steps on CPU/MPS/CUDA)
      with the prompt, by default “a high resolution, detailed photograph”.</li>
    <li><b>Ensemble</b> (<code>_ensemble_results</code>): the model outputs are resized (Lanczos-4) to the largest one and averaged with
      equal weights.</li>
    <li><b>Post-processing</b> (<code>_postprocess_image</code>): adaptive sharpening — the same unsharp mask, applied only where the
      grey image differs from its σ = 2 blur by more than 10 grey levels (edges and texture).</li>
  </ol>
  <h2>In this web app</h2>
  <div class="scroll"><table><thead><tr><th>Method</th><th>What runs</th></tr></thead><tbody>
    <tr><td>Nearest, bicubic, Lanczos-4</td><td><code>cv2.resize</code> (in the browser: <code>resample.js</code>, identical output)</td></tr>
    <tr><td>NESR pipeline, no neural models</td><td>stages 1 and 6 with the bicubic fallback in between</td></tr>
    <tr><td>Real-ESRGAN x2plus</td><td>stage 3 alone, once per ×2</td></tr>
    <tr><td>NESR pipeline with Real-ESRGAN</td><td>stages 1, 3 and 6 (no segmentation, no diffusion)</td></tr>
    <tr><td>NESR pipeline, all models</td><td>all stages (local app only; downloads the diffusion and segmentation models)</td></tr>
  </tbody></table></div>
  <div class="callout a">Fixed while building this site: the Real-ESRGAN step used to build the x2plus network with the wrong input
    layout (12 copied channels, scale 4), which returned 4× images with washed-out colours; it now runs the weights the way Real-ESRGAN
    does (3-channel input, ×2). See the SESSION_LOG for this and other fixes.</div>`;
});

route('about', async () => {
  const info = await getInfo();
  app().innerHTML = `<p class="kicker">About</p><h1>About</h1>
  <p class="lead">Neural Enhanced Super-Resolution (NESR) is a Python project for iterative image super-resolution using multiple AI
    models, with a PyQt5 desktop GUI, a command line and a Python API.</p>
  <h2>This web interface</h2>
  <p>A front end to the same Python code. ${isStatic() ? 'You are on the static showcase.' : 'You are running the local app.'}</p>
  <div class="scroll"><table><thead><tr><th></th><th>Local app</th><th>Static site</th></tr></thead><tbody>
    <tr><td>Examples (compare slider, metrics)</td><td>precomputed PNGs</td><td>precomputed PNGs</td></tr>
    <tr><td>Bicubic / Lanczos / nearest</td><td>Python (OpenCV)</td><td>in your browser (same pixels)</td></tr>
    <tr><td>PSNR / SSIM</td><td>Python</td><td>in your browser (same numbers)</td></tr>
    <tr><td>NESR pipeline, Real-ESRGAN on your image</td><td>Python, if installed</td><td>needs the local app</td></tr>
  </tbody></table></div>
  <h3>Methods on ${isStatic() ? 'the build machine' : 'this machine'}</h3>
  <ul>${list(info.methods, m => `<li><b>${esc(m.label)}</b>: ${m.available ? '<span class="ok">available</span>' : `<span class="mute">${esc(m.reason)}</span>`}</li>`)}</ul>
  <h2>Run it</h2>
  <h3>Desktop app and command line</h3>
  <pre>pip install -r requirements.txt
python main.py                                   # GUI (PyQt5)
python main.py --cli --input image.jpg --device cpu</pre>
  <h3>Web app</h3>
  <pre>pip install -r requirements-web.txt             # classical methods, pipeline without models, metrics
pip install -r requirements.txt                  # + Real-ESRGAN and diffusion
python run_web.py                                # http://127.0.0.1:8168</pre>
  <h3>Static site</h3>
  <pre>python scripts/make_examples.py                  # regenerate the examples (needs torch, basicsr, realesrgan)
python scripts/build_static.py --out site        # published by .github/workflows/pages.yml</pre>
  <h2>Credits</h2>
  <p>By George Dickinson, built with AI assistance from Claude (Anthropic); MIT licence. Real-ESRGAN by Xintao Wang et al.
    (<a href="https://github.com/xinntao/Real-ESRGAN" target="_blank" rel="noopener">xinntao/Real-ESRGAN</a>); the Stable Diffusion x4
    upscaler by Stability AI; SegFormer by NVIDIA. Source:
    <a href="https://github.com/gddickinson/neural_enhanced_super_resolution" target="_blank" rel="noopener">github.com/gddickinson/neural_enhanced_super_resolution</a>.</p>`;
});
