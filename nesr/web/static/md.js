// Small markdown renderer for the repo's README and manual: headings (with ids), paragraphs, lists, fenced code,
// tables, rules, inline code/bold/italic/links. All text is escaped; only <sub>/<sup> tags are let through.
import { esc } from './core.js';

export const slug = s => s.toLowerCase().replace(/<[^>]+>/g, '').replace(/[^\w\s-]/g, '').trim().replace(/\s+/g, '-');

function inline(s) {
  const codes = [];
  let t = esc(s).replace(/`([^`]+)`/g, (m, c) => `\u0000${codes.push(c) - 1}\u0000`);
  t = t.replace(/&lt;(\/?)(sub|sup)&gt;/g, '<$1$2>')
    .replace(/\*\*([^*]+)\*\*/g, '<b>$1</b>')
    .replace(/(^|[^*\w])\*([^*\s][^*]*?)\*(?!\w)/g, '$1<i>$2</i>')
    .replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (m, txt, href) => {
      const h = href.replace(/&amp;/g, '&');
      if (h.startsWith('#')) return `<a href="#" data-anchor="${esc(h.slice(1))}">${txt}</a>`;
      if (/^https?:\/\//.test(h)) return `<a href="${esc(h)}" target="_blank" rel="noopener">${txt}</a>`;
      return txt;
    });
  return t.replace(/\u0000(\d+)\u0000/g, (m, i) => `<code>${codes[+i]}</code>`);
}

const cells = line => line.trim().replace(/^\||\|$/g, '').split('|').map(c => c.trim());

export function markdown(src, { skipTitle = false } = {}) {
  const lines = src.replace(/\r\n/g, '\n').split('\n');
  const out = [];
  let i = 0, para = [], titleSkipped = !skipTitle;
  const flush = () => { if (para.length) { out.push(`<p>${inline(para.join(' '))}</p>`); para = []; } };
  while (i < lines.length) {
    const line = lines[i];
    if (/^```/.test(line)) {
      flush();
      const body = [];
      for (i++; i < lines.length && !/^```/.test(lines[i]); i++) body.push(lines[i]);
      out.push(`<pre><code>${esc(body.join('\n'))}</code></pre>`); i++; continue;
    }
    const h = line.match(/^(#{1,6})\s+(.*)$/);
    if (h) {
      flush();
      if (!titleSkipped && h[1].length === 1) { titleSkipped = true; i++; continue; }
      const n = Math.min(h[1].length, 4);
      out.push(`<h${n} id="md-${slug(h[2])}">${inline(h[2])}</h${n}>`); i++; continue;
    }
    if (/^\s*(---+|\*\*\*+)\s*$/.test(line)) { flush(); out.push('<hr>'); i++; continue; }
    if (/^\s*\|/.test(line) && i + 1 < lines.length && /^\s*\|?\s*:?-+/.test(lines[i + 1])) {
      flush();
      const head = cells(line), rows = [];
      for (i += 2; i < lines.length && /^\s*\|/.test(lines[i]); i++) rows.push(cells(lines[i]));
      out.push(`<div class="scroll"><table><thead><tr>${head.map(c => `<th>${inline(c)}</th>`).join('')}</tr></thead><tbody>${
        rows.map(r => `<tr>${r.map(c => `<td>${inline(c)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`);
      continue;
    }
    const li = line.match(/^(\s*)([-*]|\d+\.)\s+(.*)$/);
    if (li) {
      flush();
      const ordered = /\d/.test(li[2]), items = [];
      while (i < lines.length) {
        const m = lines[i].match(/^(\s*)([-*]|\d+\.)\s+(.*)$/);
        if (m) { items.push({ depth: m[1].length >= 2 ? 1 : 0, text: m[3] }); i++; continue; }
        if (/^\s{2,}\S/.test(lines[i]) && items.length && !/^\s*```/.test(lines[i])) { items[items.length - 1].text += ' ' + lines[i].trim(); i++; continue; }
        break;
      }
      const tag = ordered ? 'ol' : 'ul';
      out.push(`<${tag}>${items.map(it => `<li${it.depth ? ' style="margin-left:1.2em"' : ''}>${inline(it.text)}</li>`).join('')}</${tag}>`);
      continue;
    }
    if (!line.trim()) { flush(); i++; continue; }
    para.push(line.trim()); i++;
  }
  flush();
  return out.join('\n');
}

// In-page anchors ([text](#section)) scroll to the matching heading instead of changing the route.
export function wireAnchors(root) {
  root.addEventListener('click', e => {
    const a = e.target.closest('a[data-anchor]');
    if (!a) return;
    e.preventDefault();
    const want = a.dataset.anchor.replace(/^\d+-?/, '');
    const h = [...root.querySelectorAll('h1,h2,h3,h4')].find(x => x.id === 'md-' + a.dataset.anchor || x.id.replace(/^md-\d+-?/, '') === want);
    if (h) h.scrollIntoView({ behavior: 'smooth', block: 'start' });
  });
}
