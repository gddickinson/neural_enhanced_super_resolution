// Before / after comparison: two images of the same size stacked, the right one clipped at a divider that follows
// the pointer (drag or tap) or the labelled range input (keyboard). Zoom shows real pixels (no smoothing).
import { esc } from './core.js';
import { toCanvas } from './images.js';

export function beforeAfter(host, { left, right, leftLabel, rightLabel, zoom = 1, pos = 50 }) {
  host.innerHTML = `<div class="ba-wrap"><div class="ba" style="width:${zoom * 100}%">
      <span class="lbl l"></span><div class="clip"></div><span class="lbl r"></span><div class="bar"></div></div></div>
    <label for="ba-pos">Divider position <b class="mono"></b></label>
    <input id="ba-pos" type="range" min="0" max="100" step="0.5" value="${pos}">`;
  const ba = host.querySelector('.ba'), clip = ba.querySelector('.clip'), bar = ba.querySelector('.bar');
  const input = host.querySelector('#ba-pos'), out = host.querySelector('label b');
  const cl = toCanvas(left), cr = toCanvas(right);
  cl.setAttribute('aria-label', leftLabel); cr.setAttribute('aria-label', rightLabel);
  cl.setAttribute('role', 'img'); cr.setAttribute('role', 'img');
  ba.insertBefore(cl, ba.firstChild); clip.appendChild(cr);
  ba.querySelector('.lbl.l').innerHTML = esc(leftLabel);
  ba.querySelector('.lbl.r').innerHTML = esc(rightLabel);
  const set = p => {
    p = Math.max(0, Math.min(100, p));
    clip.style.clipPath = `inset(0 0 0 ${p}%)`;
    bar.style.left = p + '%';
    input.value = p; out.textContent = `${Math.round(p)}%`;
  };
  const fromEvent = e => { const r = ba.getBoundingClientRect(); set(((e.clientX - r.left) / r.width) * 100); };
  let dragging = false;
  ba.addEventListener('pointerdown', e => { dragging = true; ba.setPointerCapture(e.pointerId); fromEvent(e); });
  ba.addEventListener('pointermove', e => { if (dragging) fromEvent(e); });
  ba.addEventListener('pointerup', () => { dragging = false; });
  input.addEventListener('input', () => set(+input.value));
  set(pos);
  return { set, get: () => +input.value };
}
