// Node test: resample.js reproduces cv2.resize (tests/fixtures/js_cases.json, written by scripts/make_js_fixtures.py).
// 8-bit: identical pixels; float32: within 1e-5. Run: node nesr/web/static/resample.test.mjs
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resize, resizeFloat } from './resample.js';

const fx = JSON.parse(readFileSync(new URL('../../../tests/fixtures/js_cases.json', import.meta.url), 'utf8'));
let values = 0;
for (const c of fx.resize) {
  const i = fx.inputs[c.input];
  const out = resize(Uint8Array.from(i.data), i.w, i.h, i.cn, c.out_w, c.out_h, c.method);
  assert.equal(out.length, c.out.length, `${c.input} ${c.method} size`);
  for (let k = 0; k < out.length; k++) {
    if (out[k] !== c.out[k]) assert.fail(`${c.input} ${c.method} ${c.out_w}x${c.out_h}: value ${k} is ${out[k]}, OpenCV ${c.out[k]}`);
  }
  values += out.length;
}
const F = fx.resize_float;
for (const c of F.cases) {
  const out = resizeFloat(Float64Array.from(F.data), F.w, F.h, F.cn, c.out_w, c.out_h, c.method);
  out.forEach((v, k) => assert.ok(Math.abs(v - c.out[k]) < 1e-5, `float ${c.method} ${c.out_w}x${c.out_h} value ${k}: ${v} vs ${c.out[k]}`));
  values += out.length;
}
// RGBA input (canvas pixels): colour channels as for RGB, alpha forced to 255.
const rgb = fx.inputs.zone_24, rgba = new Uint8Array(rgb.w * rgb.h * 4);
for (let p = 0; p < rgb.w * rgb.h; p++) { rgba.set(rgb.data.slice(p * 3, p * 3 + 3), p * 4); rgba[p * 4 + 3] = 255; }
const a3 = resize(Uint8Array.from(rgb.data), rgb.w, rgb.h, 3, 48, 48, 'lanczos'), a4 = resize(rgba, rgb.w, rgb.h, 4, 48, 48, 'lanczos');
for (let p = 0; p < 48 * 48; p++) { for (let c = 0; c < 3; c++) assert.equal(a4[p * 4 + c], a3[p * 3 + c]); assert.equal(a4[p * 4 + 3], 255); }
assert.throws(() => resize(new Uint8Array(3), 1, 1, 3, 2, 2, 'bilinear'), /unknown method/);
console.log(`resample.js: ${fx.resize.length} 8-bit and ${F.cases.length} float cases, ${values} values match OpenCV ${fx.opencv}`);
