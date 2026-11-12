// Image helpers: load a URL or File into RGBA pixels, draw pixels to a canvas, upscale with resample.js.
// An "image" here is { w, h, data } with data a Uint8ClampedArray of w * h * 4 RGBA values.
import { resize } from './resample.js';

export async function loadUrl(url) {
  const img = new Image();
  img.decoding = 'async';
  img.src = url;
  await img.decode();
  return fromDrawable(img, img.naturalWidth, img.naturalHeight);
}

export async function loadFile(file) {
  const bmp = await createImageBitmap(file, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
  const r = fromDrawable(bmp, bmp.width, bmp.height);
  bmp.close();
  return r;
}

function fromDrawable(src, w, h) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const g = c.getContext('2d', { willReadFrequently: true, colorSpace: 'srgb' });
  g.drawImage(src, 0, 0);
  const data = g.getImageData(0, 0, w, h).data;
  for (let i = 3; i < data.length; i += 4) data[i] = 255; // alpha is ignored (flattened to opaque)
  return { w, h, data };
}

export function toCanvas(im, canvas = document.createElement('canvas')) {
  canvas.width = im.w; canvas.height = im.h;
  canvas.getContext('2d').putImageData(new ImageData(new Uint8ClampedArray(im.data), im.w, im.h), 0, 0);
  return canvas;
}

export const toDataUrl = im => toCanvas(im).toDataURL('image/png');

// Upscale to an exact size (or by a factor) with nearest / bicubic / lanczos (OpenCV-compatible, see resample.js).
export function upscale(im, method, outW, outH = outW) {
  return { w: outW, h: outH, data: resize(im.data, im.w, im.h, 4, outW, outH, method) };
}

// Fraction of RGB values identical in two same-size images, and the largest difference.
export function compare(a, b) {
  if (a.w !== b.w || a.h !== b.h) return { same: 0, maxDiff: Infinity };
  let same = 0, n = 0, maxDiff = 0;
  for (let i = 0; i < a.data.length; i++) {
    if ((i & 3) === 3) continue;
    const d = Math.abs(a.data[i] - b.data[i]);
    same += d === 0; n++; if (d > maxDiff) maxDiff = d;
  }
  return { same: same / n, maxDiff };
}

export const fmtPsnr = v => (v === null || v === undefined ? '—' : !Number.isFinite(v) ? '∞' : v.toFixed(2));
export const fmtSsim = v => (v === null || v === undefined ? '—' : v.toFixed(4));
