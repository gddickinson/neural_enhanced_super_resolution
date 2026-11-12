// Static-site data layer: answers api() calls from the JSON exported by scripts/build_static.py, and runs the
// classical upscalers (resample.js) and metrics (metrics.js) in the browser for /upscale. Model methods need the
// local app.
import { loadFile, loadUrl, toDataUrl, upscale } from './images.js';
import { psnr, ssim } from './metrics.js';
import { resize } from './resample.js';

const files = new Map();
async function file(rel) {
  if (!files.has(rel)) files.set(rel, fetch('api/' + rel + '.json').then(r => { if (!r.ok) throw new Error(`${r.status} ${rel}`); return r.json(); }));
  return files.get(rel);
}

export const NEEDS_SERVER = 'needs the full app running locally (Python server): see the README, "Run the web app"';
const BROWSER = { nearest: 'Nearest neighbour', bicubic: 'Bicubic', lanczos: 'Lanczos-4' };
const MAX_SIDE = 2048;

export async function staticApi(path, body) {
  const clean = path.split('?')[0].replace(/^\/+/, '');
  if (clean === 'upscale') return upscaleInBrowser(body);
  return file(clean);
}

async function upscaleInBrowser(fd) {
  const t0 = performance.now();
  const method = fd.get('method'), scale = +fd.get('scale'), mode = fd.get('mode');
  if (!BROWSER[method]) throw new Error(`This method ${NEEDS_SERVER}. Nearest, bicubic and Lanczos run in your browser.`);
  const f = fd.get('file');
  let img, source;
  if (f && f.name) { img = await loadFile(f); source = f.name; } else {
    const p = fd.get('pattern');
    img = await loadUrl(`static/examples/${p}_hr.png`); source = `${p} (256 px)`;
  }
  let lr = img, ref = null;
  if (mode === 'benchmark') {
    const w = img.w - (img.w % scale), h = img.h - (img.h % scale);
    ref = crop(img, w, h);
    if (Math.max(w, h) > MAX_SIDE * scale) throw new Error(`image too large (longest side ${MAX_SIDE * scale} px in benchmark mode)`);
    const lw = Math.trunc(w / scale), lh = Math.trunc(h / scale);
    if (Math.min(lw, lh) < 1) throw new Error('image too small for this factor');
    lr = { w: lw, h: lh, data: resize(ref.data, w, h, 4, lw, lh, 'bicubic') };
  } else if (Math.max(img.w, img.h) > MAX_SIDE) throw new Error(`image too large (longest side ${MAX_SIDE} px in the browser)`);
  const out = upscale(lr, method, lr.w * scale, lr.h * scale);
  const r = { mode, source, input_size: [lr.w, lr.h], output_size: [out.w, out.h], input: toDataUrl(lr), output: toDataUrl(out),
    info: { method, label: BROWSER[method], scale, where: 'computed in your browser', seconds: +((performance.now() - t0) / 1000).toFixed(2) } };
  if (ref) {
    r.reference = toDataUrl(ref);
    r.metrics = { psnr: psnr(ref.data, out.data, ref.w, ref.h, 4) };
    r.metrics.ssim = Math.min(ref.w, ref.h) >= 11 ? ssim(ref.data, out.data, ref.w, ref.h, 4) : null;
  }
  return r;
}

function crop(im, w, h) {
  if (w === im.w && h === im.h) return im;
  const data = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) data.set(im.data.subarray(y * im.w * 4, y * im.w * 4 + w * 4), y * w * 4);
  return { w, h, data };
}
