// Entry point: lazy page modules, the mode label, a subtle "resolving pixels" background, then the router.
import { $, app, isStatic, lazy, route, start } from './core.js';

const PAGES = { 'home.js': ['home'], 'compare.js': ['compare'], 'upscale.js': ['upscale'], 'metrics_page.js': ['metrics'],
  'docs.js': ['pipeline', 'readme', 'usage', 'structure', 'about'] };
lazy(Object.fromEntries(Object.entries(PAGES).flatMap(([file, names]) => names.map(n => [n, () => import('./' + file)]))));
route('notfound', () => { app().innerHTML = '<h1>Not found</h1><p><a href="#/">Back to the start</a></p>'; });

$('#mode').textContent = isStatic()
  ? 'static site: model outputs precomputed by the Python code; bicubic, Lanczos and metrics run in your browser'
  : 'local app: upscaling runs in Python on this computer';

// Background: a few faint squares that split into finer blocks and merge again. Off under prefers-reduced-motion.
function background(canvas) {
  if (!canvas || matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  const g = canvas.getContext('2d');
  const fit = () => { canvas.width = innerWidth; canvas.height = innerHeight; };
  fit(); addEventListener('resize', fit);
  const cols = ['#3ee8ff', '#ff9f2e', '#a67bff'];
  const tiles = Array.from({ length: 9 }, (_, i) => ({ x: Math.random(), y: Math.random(), s: 60 + Math.random() * 80,
    c: cols[i % 3], p: Math.random() * 6.28, v: 0.0004 + Math.random() * 0.0004 }));
  let last = 0;
  const tick = t => {
    requestAnimationFrame(tick);
    if (document.hidden || t - last < 80) return;
    last = t;
    g.clearRect(0, 0, canvas.width, canvas.height);
    for (const q of tiles) {
      const n = 2 ** (1 + Math.round(2.5 + 2.5 * Math.sin(q.p + t * q.v))), step = q.s / n;
      const x0 = q.x * canvas.width, y0 = q.y * canvas.height;
      g.strokeStyle = q.c; g.globalAlpha = 0.2; g.lineWidth = 1;
      for (let i = 0; i <= n; i++) {
        g.beginPath(); g.moveTo(x0 + i * step, y0); g.lineTo(x0 + i * step, y0 + q.s); g.stroke();
        g.beginPath(); g.moveTo(x0, y0 + i * step); g.lineTo(x0 + q.s, y0 + i * step); g.stroke();
      }
    }
    g.globalAlpha = 1;
  };
  requestAnimationFrame(tick);
}
background($('#bg'));
start();
