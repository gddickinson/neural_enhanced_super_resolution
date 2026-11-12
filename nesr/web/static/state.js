// Shared cached data: project info and the precomputed examples manifest.
import { api } from './core.js';

export const getInfo = () => api('/info');
export const getExamples = () => api('/examples');
export const exampleUrl = file => 'static/examples/' + file;

// Methods that can run in the current context (static site: only the browser ports).
export function runnable(info, staticSite) {
  return info.methods.map(m => ({ ...m, ok: staticSite ? m.browser : m.available,
    why: staticSite && !m.browser ? 'needs the local app' : m.reason }));
}
