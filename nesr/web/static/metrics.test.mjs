// Node test: metrics.js reproduces nesr/metrics.py (tests/fixtures/js_cases.json). Run: node nesr/web/static/metrics.test.mjs
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { psnr, ssim } from './metrics.js';

const fx = JSON.parse(readFileSync(new URL('../../../tests/fixtures/js_cases.json', import.meta.url), 'utf8'));
const close = (a, b, what) => assert.ok(Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(b)), `${what}: ${a} vs ${b}`);
for (const c of fx.metrics) {
  const p = psnr(c.ref, c.test, c.w, c.h, c.cn), s = ssim(c.ref, c.test, c.w, c.h, c.cn);
  if (c.psnr === null) assert.equal(p, Infinity, `${c.name} psnr`); else close(p, c.psnr, `${c.name} psnr`);
  close(s, c.ssim, `${c.name} ssim`);
  // the same pair as RGBA (canvas layout) gives the same numbers
  if (c.cn === 3) {
    const four = arr => { const o = new Uint8Array(c.w * c.h * 4); for (let i = 0; i < c.w * c.h; i++) { o.set(arr.slice(i * 3, i * 3 + 3), i * 4); o[i * 4 + 3] = 255; } return o; };
    close(ssim(four(c.ref), four(c.test), c.w, c.h, 4), c.ssim, `${c.name} ssim rgba`);
  }
}
assert.throws(() => ssim(new Uint8Array(27), new Uint8Array(27), 3, 3, 3), /at least 11/);
console.log(`metrics.js: ${fx.metrics.length} image pairs match nesr/metrics.py (PSNR and SSIM within 1e-9)`);
