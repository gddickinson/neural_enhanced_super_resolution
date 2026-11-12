// Image quality metrics, a port of nesr/metrics.py (tested against it by metrics.test.mjs):
// PSNR in dB over all pixels and colour channels, and SSIM (Wang et al. 2004) with an 11x11 Gaussian window
// (sigma 1.5), K1 = 0.01, K2 = 0.03, mirrored borders, averaged over the image minus a 5-px border and over channels.
// Images are interleaved arrays (w * h * cn); with cn = 4 the alpha channel is ignored.

export const SIGMA = 1.5, RADIUS = 5, K1 = 0.01, K2 = 0.03;

function colourChannels(cn) { return cn === 4 ? 3 : cn; }

function checkPair(a, b, w, h, cn) {
  if (a.length !== w * h * cn || b.length !== w * h * cn) throw new Error('images must both be w x h x channels');
}

export function psnr(a, b, w, h, cn, dataRange = 255) {
  checkPair(a, b, w, h, cn);
  const ch = colourChannels(cn);
  let se = 0;
  for (let i = 0; i < w * h; i++) for (let c = 0; c < ch; c++) { const d = a[i * cn + c] - b[i * cn + c]; se += d * d; }
  const mse = se / (w * h * ch);
  return mse === 0 ? Infinity : 10 * Math.log10(dataRange * dataRange / mse);
}

export function gaussianKernel(sigma = SIGMA, radius = RADIUS) {
  const k = [];
  let s = 0;
  for (let x = -radius; x <= radius; x++) { const v = Math.exp(-0.5 * (x / sigma) ** 2); k.push(v); s += v; }
  return k.map(v => v / s);
}

// 'reflect' border: d c b a | a b c d | d c b a
function reflect(i, n) {
  const p = 2 * n;
  i = ((i % p) + p) % p;
  return i >= n ? p - 1 - i : i;
}

// Separable Gaussian blur of a w x h Float64Array with mirrored borders.
export function gaussianFilter(img, w, h, k = gaussianKernel()) {
  const r = (k.length - 1) / 2, tmp = new Float64Array(w * h), out = new Float64Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let s = 0;
    for (let i = -r; i <= r; i++) s += k[i + r] * img[y * w + reflect(x + i, w)];
    tmp[y * w + x] = s;
  }
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let s = 0;
    for (let i = -r; i <= r; i++) s += k[i + r] * tmp[reflect(y + i, h) * w + x];
    out[y * w + x] = s;
  }
  return out;
}

function plane(a, w, h, cn, c) {
  const p = new Float64Array(w * h);
  for (let i = 0; i < w * h; i++) p[i] = a[i * cn + c];
  return p;
}

// Per-pixel SSIM of two single-channel Float64Arrays.
export function ssimMap(x, y, w, h, dataRange = 255) {
  const c1 = (K1 * dataRange) ** 2, c2 = (K2 * dataRange) ** 2, n = w * h, k = gaussianKernel();
  const xx = new Float64Array(n), yy = new Float64Array(n), xy = new Float64Array(n);
  for (let i = 0; i < n; i++) { xx[i] = x[i] * x[i]; yy[i] = y[i] * y[i]; xy[i] = x[i] * y[i]; }
  const ux = gaussianFilter(x, w, h, k), uy = gaussianFilter(y, w, h, k);
  const uxx = gaussianFilter(xx, w, h, k), uyy = gaussianFilter(yy, w, h, k), uxy = gaussianFilter(xy, w, h, k);
  const m = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const vx = uxx[i] - ux[i] * ux[i], vy = uyy[i] - uy[i] * uy[i], vxy = uxy[i] - ux[i] * uy[i];
    m[i] = ((2 * ux[i] * uy[i] + c1) * (2 * vxy + c2)) / ((ux[i] * ux[i] + uy[i] * uy[i] + c1) * (vx + vy + c2));
  }
  return m;
}

export function ssim(a, b, w, h, cn, dataRange = 255) {
  checkPair(a, b, w, h, cn);
  if (Math.min(w, h) < 2 * RADIUS + 1) throw new Error(`SSIM needs images of at least ${2 * RADIUS + 1} x ${2 * RADIUS + 1} pixels`);
  const ch = colourChannels(cn), r = RADIUS;
  let total = 0;
  for (let c = 0; c < ch; c++) {
    const m = ssimMap(plane(a, w, h, cn, c), plane(b, w, h, cn, c), w, h, dataRange);
    let s = 0;
    for (let y = r; y < h - r; y++) for (let x = r; x < w - r; x++) s += m[y * w + x];
    total += s / ((w - 2 * r) * (h - 2 * r));
  }
  return total / ch;
}
