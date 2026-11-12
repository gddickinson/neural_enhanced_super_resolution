// Classical resampling, a port of OpenCV's cv2.resize (the call the NESR pipeline uses for its bicubic fallback,
// its ensemble alignment and the degradation tool): INTER_NEAREST, INTER_CUBIC (a = -0.75) and INTER_LANCZOS4
// (8 taps), pixel-centre alignment, replicated borders. (INTER_LINEAR is left out: OpenCV switches it to other
// kernels for some sizes, e.g. area averaging for an exact 2x reduction.)
// resize() reproduces OpenCV's 8-bit fixed-point arithmetic (11-bit coefficients, rounded 22-bit shift), so its
// output matches cv2.resize on uint8 images; resizeFloat() follows OpenCV's float32 path in double precision.
// Tested against Python/OpenCV by resample.test.mjs (fixture: scripts/make_js_fixtures.py).

export const METHODS = {
  nearest: { label: 'Nearest neighbour', ksize: 0 },
  bicubic: { label: 'Bicubic (OpenCV, a = -0.75)', ksize: 4 },
  lanczos: { label: 'Lanczos-4 (8 taps)', ksize: 8 },
};

const f = Math.fround;
const COEF_BITS = 11, COEF_SCALE = 1 << COEF_BITS;

// cvRound: round half to even (float -> int).
function cvRound(v) {
  const r = Math.round(v);
  return (Math.abs(v % 1) === 0.5 && r % 2 !== 0) ? r - 1 : r;
}

function cubicCoeffs(x) {
  const A = f(-0.75), x1 = f(x + 1), ox = f(1 - x);
  const c0 = f(f(f(f(f(f(A * x1) - f(5 * A)) * x1) + f(8 * A)) * x1) - f(4 * A));
  const c1 = f(f(f(f(f(f(A + 2) * x) - f(A + 3)) * x) * x) + 1);
  const c2 = f(f(f(f(f(f(A + 2) * ox) - f(A + 3)) * ox) * ox) + 1);
  return [c0, c1, c2, f(f(f(1 - c0) - c1) - c2)];
}

const S45 = 0.70710678118654752440084436210485;
const CS = [[1, 0], [-S45, -S45], [0, 1], [S45, -S45], [-1, 0], [S45, S45], [0, -1], [-S45, S45]];
function lanczosCoeffs(x) {
  if (x < 1.1920928955078125e-7) return [0, 0, 0, 1, 0, 0, 0, 0];
  const y0 = -(x + 3) * Math.PI * 0.25, s0 = Math.sin(y0), c0 = Math.cos(y0);
  const c = new Array(8);
  let sum = 0;
  for (let i = 0; i < 8; i++) {
    const y = -(x + 3 - i) * Math.PI * 0.25;
    c[i] = f((CS[i][0] * s0 + CS[i][1] * c0) / (y * y));
    sum = f(sum + c[i]);
  }
  sum = f(1 / sum);
  return c.map(v => f(v * sum));
}

// Per output index: first source index (before clamping) and the tap weights, as OpenCV computes them.
function taps(outN, inN, method) {
  const scale = 1 / (outN / inN), ksize = METHODS[method].ksize, k2 = ksize / 2;
  const ofs = new Int32Array(outN), w = [];
  for (let d = 0; d < outN; d++) {
    let fx = f((d + 0.5) * scale - 0.5), sx = Math.floor(fx);
    fx = f(fx - sx);
    w.push(method === 'bicubic' ? cubicCoeffs(fx) : lanczosCoeffs(fx));
    ofs[d] = sx - k2 + 1;
  }
  return { ofs, w, ksize };
}

const clampI = (i, n) => (i < 0 ? 0 : i >= n ? n - 1 : i);

function nearest(src, w, h, cn, outW, outH, Out) {
  const out = new Out(outW * outH * cn), fx = w / outW, fy = h / outH;
  for (let y = 0; y < outH; y++) {
    const sy = Math.min(Math.floor(y * fy), h - 1);
    for (let x = 0; x < outW; x++) {
      const sx = Math.min(Math.floor(x * fx), w - 1);
      for (let c = 0; c < cn; c++) out[(y * outW + x) * cn + c] = src[(sy * w + sx) * cn + c];
    }
  }
  return out;
}

// Vertical 8-bit pass as OpenCV's SIMD kernels compute it (they cover all but the last few values of a row):
// bicubic in float32 with fused multiply-adds, Lanczos-4 in integer fixed point.
function vertical(method, r, b, half, div) {
  if (method === 'bicubic') {
    const sc = f(1 / (COEF_SCALE * COEF_SCALE)), bb = b.map(v => f(v * sc));
    const s = f(r(0) * bb[0] + f(r(1) * bb[1] + f(r(2) * bb[2] + f(r(3) * bb[3]))));
    return cvRound(s);
  }
  let s = 0;
  for (let k = 0; k < b.length; k++) s += r(k) * b[k];
  return Math.floor((s + half) / div);
}

// Resize an 8-bit interleaved image (cn channels; with cn = 4 the alpha channel is set to 255).
export function resize(src, w, h, cn, outW, outH, method = 'bicubic') {
  check(src, w, h, cn, outW, outH, method);
  if (method === 'nearest') return nearest(src, w, h, cn, outW, outH, Uint8ClampedArray);
  const X = taps(outW, w, method), Y = taps(outH, h, method), K = X.ksize, ch = cn === 4 ? 3 : cn;
  const ax = X.w.map(ws => ws.map(v => cvRound(v * COEF_SCALE)));
  const ay = Y.w.map(ws => ws.map(v => cvRound(v * COEF_SCALE)));
  // Horizontal pass on every source row (integers), then vertical pass with a rounded 22-bit shift.
  const rows = new Float64Array(h * outW * ch);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < outW; x++) {
      const o = X.ofs[x], a = ax[x];
      for (let c = 0; c < ch; c++) {
        let s = 0;
        for (let k = 0; k < K; k++) s += src[(y * w + clampI(o + k, w)) * cn + c] * a[k];
        rows[(y * outW + x) * ch + c] = s;
      }
    }
  }
  const out = new Uint8ClampedArray(outW * outH * cn), half = 2 ** (2 * COEF_BITS - 1), div = 2 ** (2 * COEF_BITS);
  for (let y = 0; y < outH; y++) {
    const o = Y.ofs[y], b = ay[y];
    for (let x = 0; x < outW; x++) {
      for (let c = 0; c < ch; c++) {
        const r = k => rows[(clampI(o + k, h) * outW + x) * ch + c];
        out[(y * outW + x) * cn + c] = vertical(method, r, b, half, div);
      }
      if (cn === 4) out[(y * outW + x) * 4 + 3] = 255;
    }
  }
  return out;
}

// Resize a float image (values in any range) the way OpenCV does for float32 input; returns a Float64Array.
export function resizeFloat(src, w, h, cn, outW, outH, method = 'bicubic') {
  check(src, w, h, cn, outW, outH, method);
  if (method === 'nearest') return nearest(src, w, h, cn, outW, outH, Float64Array);
  const X = taps(outW, w, method), Y = taps(outH, h, method), K = X.ksize;
  const rows = new Float64Array(h * outW * cn), out = new Float64Array(outW * outH * cn);
  for (let y = 0; y < h; y++) for (let x = 0; x < outW; x++) for (let c = 0; c < cn; c++) {
    let s = 0;
    for (let k = 0; k < K; k++) s += src[(y * w + clampI(X.ofs[x] + k, w)) * cn + c] * X.w[x][k];
    rows[(y * outW + x) * cn + c] = s;
  }
  for (let y = 0; y < outH; y++) for (let x = 0; x < outW; x++) for (let c = 0; c < cn; c++) {
    let s = 0;
    for (let k = 0; k < K; k++) s += rows[(clampI(Y.ofs[y] + k, h) * outW + x) * cn + c] * Y.w[y][k];
    out[(y * outW + x) * cn + c] = s;
  }
  return out;
}

function check(src, w, h, cn, outW, outH, method) {
  if (!METHODS[method]) throw new Error(`unknown method ${method}`);
  if (!(w >= 1 && h >= 1 && outW >= 1 && outH >= 1)) throw new Error('image sizes must be at least 1 px');
  if (src.length !== w * h * cn) throw new Error(`expected ${w * h * cn} values, got ${src.length}`);
}
