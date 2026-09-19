/**
 * The monitoring page, served by the control service itself.
 *
 * Deliberately one self-contained file with no build step and no framework: it is
 * operational tooling that must work when other things are broken, and a dashboard that
 * needs a toolchain to run is a dashboard you cannot trust in a crisis.
 */
export const dashboardHtml = (): string => `<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fleet</title>
<style>
  :root{
    --bg:#f4f6f7; --card:#fff; --ink:#10161b; --ink2:#46555f; --ink3:#7b8791;
    --rule:#dde3e7; --accent:#0b6e8c; --ok:#2c6b45; --okbg:#e2f0e8;
    --warn:#8d5310; --warnbg:#f7ecdd; --off:#8a949c; --offbg:#eceff1; --bad:#8c2f2f; --badbg:#f7e4e4;
  }
  @media (prefers-color-scheme:dark){:root{
    --bg:#0e1316; --card:#161c21; --ink:#e4eaed; --ink2:#a9b6bf; --ink3:#78858e;
    --rule:#28333a; --accent:#4bb2d1; --ok:#6cbb8c; --okbg:#15251c;
    --warn:#d6a25c; --warnbg:#2a2115; --off:#7d8891; --offbg:#1c2429; --bad:#d98686; --badbg:#2a1717;
  }}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
    font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;padding:20px 16px 60px}
  main{max-width:1100px;margin:0 auto}
  header{display:flex;flex-wrap:wrap;gap:10px 18px;align-items:baseline;margin:0 0 20px}
  h1{font-size:19px;margin:0;letter-spacing:-.01em}
  .addr{font:12px ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--ink3);word-break:break-all}
  .live{margin-left:auto;font-size:12px;color:var(--ink3);display:flex;align-items:center;gap:6px}
  .dot{width:7px;height:7px;border-radius:50%;background:var(--ok)}
  @media (prefers-reduced-motion:no-preference){.dot{animation:p 2s ease-in-out infinite}}
  @keyframes p{50%{opacity:.25}}
  h2{font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--ink3);
    margin:28px 0 10px;font-weight:600}
  .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px}
  .card{background:var(--card);border:1px solid var(--rule);border-radius:8px;padding:14px 16px}
  .host{display:flex;flex-direction:column;gap:8px}
  .host .top{display:flex;align-items:start;gap:10px}
  .name{font-weight:600;font-size:14px;overflow-wrap:anywhere;line-height:1.35}
  .pill{font:11px ui-monospace,monospace;padding:2px 7px;border-radius:99px;white-space:nowrap;flex:none}
  .on{color:var(--ok);background:var(--okbg)} .offp{color:var(--off);background:var(--offbg)}
  .pausedp{color:var(--warn);background:var(--warnbg)}
  .spec{font-size:12.5px;color:var(--ink2);display:flex;flex-wrap:wrap;gap:3px 12px}
  .bar{height:4px;background:var(--offbg);border-radius:99px;overflow:hidden}
  .bar>i{display:block;height:100%;background:var(--accent);border-radius:99px;transition:width .4s ease}
  table{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}
  th{text-align:left;font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink3);
    padding:8px 10px;border-bottom:1px solid var(--rule);font-weight:600;white-space:nowrap}
  td{padding:8px 10px;border-bottom:1px solid var(--rule);color:var(--ink2);vertical-align:top}
  tr:last-child td{border-bottom:0}
  td.n{text-align:right;font-family:ui-monospace,monospace}
  .tw{overflow-x:auto;background:var(--card);border:1px solid var(--rule);border-radius:8px}
  .st{font:11px ui-monospace,monospace;padding:1px 6px;border-radius:99px}
  .s-succeeded{color:var(--ok);background:var(--okbg)}
  .s-running,.s-leased,.s-offered{color:var(--accent);background:var(--offbg)}
  .s-failed{color:var(--bad);background:var(--badbg)}
  .s-pending{color:var(--ink3);background:var(--offbg)}
  .s-cancelled{color:var(--warn);background:var(--warnbg)}
  .ev{font:12px ui-monospace,monospace;color:var(--ink2);padding:5px 10px;border-bottom:1px solid var(--rule);
    display:flex;gap:10px;white-space:nowrap;overflow-x:auto}
  .ev b{color:var(--ink);font-weight:600}
  .ev .t{color:var(--ink3);flex:none}
  .muted{color:var(--ink3);font-size:13px;padding:14px 16px}
  .kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:12px}
  .kpi{background:var(--card);border:1px solid var(--rule);border-radius:8px;padding:12px 14px}
  .kpi .v{font:600 22px/1.15 ui-monospace,monospace;letter-spacing:-.02em}
  .kpi .l{font-size:11px;letter-spacing:.07em;text-transform:uppercase;color:var(--ink3);margin-top:3px}
  .err{color:var(--bad)}
  .now{background:var(--card);border:1px solid var(--rule);border-left:3px solid var(--accent);
    border-radius:8px;padding:16px 18px;margin:20px 0 0}
  .now h3{margin:0 0 4px;font-size:15px}
  .now .meta{font-size:12.5px;color:var(--ink3);margin-bottom:12px}
  .bigbar{height:10px;background:var(--offbg);border-radius:99px;overflow:hidden;margin:10px 0 6px}
  .bigbar>i{display:block;height:100%;background:var(--accent);border-radius:99px;transition:width .5s ease}
  .legend{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:12.5px;color:var(--ink2);
    font-variant-numeric:tabular-nums}
  .legend b{color:var(--ink)}
  .inflight{font-size:12px;color:var(--accent)}
</style></head><body><main>

<header>
  <h1>Fleet</h1>
  <span class="addr" id="addr"></span>
  <span class="live"><span class="dot"></span><span id="tick">connecting…</span></span>
</header>

<div class="kpis" id="kpis"></div>

<div id="current"></div>

<h2>Computers</h2>
<div class="grid" id="hosts"></div>

<h2>Recent jobs</h2>
<div class="tw"><table><thead><tr>
  <th>Job</th><th>Type</th><th>Status</th><th class="n">Done</th><th class="n">Items</th>
  <th>Split across computers</th><th class="n">Started</th>
</tr></thead><tbody id="jobs"></tbody></table></div>

<h2>Activity</h2>
<div class="tw" id="events"></div>

</main><script>
const $ = id => document.getElementById(id)
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))
const ago = iso => {
  if (!iso) return '—'
  const s = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000)
  if (s < 60) return Math.floor(s) + 's ago'
  if (s < 3600) return Math.floor(s / 60) + 'm ago'
  return Math.floor(s / 3600) + 'h ago'
}

async function get (path) {
  const res = await fetch(path, { credentials: 'same-origin' })
  if (!res.ok) throw new Error(path + ' -> ' + res.status)
  return res.json()
}

function renderHosts (hosts) {
  if (!hosts.length) { $('hosts').innerHTML = '<div class="card muted">No computers enrolled yet.</div>'; return }
  $('hosts').innerHTML = hosts.map(h => {
    const state = h.revoked_at ? ['offp','revoked'] : h.paused ? ['pausedp','paused']
      : h.online ? ['on','online'] : ['offp','offline']
    return '<div class="card host">' +
      '<div class="top"><span class="name">' + esc(h.label) + '</span>' +
      '<span class="pill ' + state[0] + '">' + state[1] + '</span></div>' +
      '<div class="spec">' +
        '<span>' + esc(h.os ?? '?') + '/' + esc(h.arch ?? '?') + '</span>' +
        '<span>' + (h.logical_cores ?? '?') + ' cores</span>' +
        '<span>' + (h.total_ram_mb ? Math.round(h.total_ram_mb / 1024) + ' GB' : '?') + '</span>' +
        '<span>seen ' + ago(h.last_heartbeat_at) + '</span>' +
      '</div></div>'
  }).join('')
}

function renderCurrent (jobs) {
  const live = jobs.find(j => j.status === 'running')
  if (!live) {
    const last = jobs[0]
    $('current').innerHTML = last
      ? '<div class="now"><h3>No job running</h3><div class="meta">Last: ' + esc(last.adapter) +
        ' — ' + esc(last.status) + ', ' + last.done + '/' + last.total_items + ' ' + ago(last.created_at) + '</div></div>'
      : ''
    return
  }
  const pct = live.total_items ? (100 * live.done / live.total_items) : 0
  const split = (live.split ?? []).map(s =>
    '<span><b>' + esc(s.label.slice(0, 26)) + '</b> ' + s.n + ' slices</span>').join('')
  $('current').innerHTML =
    '<div class="now"><h3>Running now — ' + esc(live.adapter) + '</h3>' +
    '<div class="meta">job ' + esc(live.id.slice(0, 8)) + ' · started ' + ago(live.created_at) + '</div>' +
    '<div class="bigbar"><i style="width:' + pct.toFixed(1) + '%"></i></div>' +
    '<div class="legend"><span><b>' + live.done + '</b> of ' + live.total_items + ' slices (' +
    pct.toFixed(0) + '%)</span>' +
    (live.inFlight ? '<span class="inflight">' + live.inFlight + ' in flight</span>' : '') +
    split + '</div></div>'
}

function renderJobs (jobs) {
  if (!jobs.length) { $('jobs').innerHTML = '<tr><td colspan="7" class="muted">No jobs yet.</td></tr>'; return }
  $('jobs').innerHTML = jobs.map(j => {
    const pct = j.total_items ? Math.round(100 * j.done / j.total_items) : 0
    const split = (j.split ?? []).map(s => esc(s.label) + ' ' + s.n).join(' · ') || '—'
    return '<tr>' +
      '<td><code>' + esc(j.id.slice(0, 8)) + '</code></td>' +
      '<td>' + esc(j.adapter) + '</td>' +
      '<td><span class="st s-' + esc(j.status) + '">' + esc(j.status) + '</span></td>' +
      '<td class="n">' + j.done + '<div class="bar" style="margin-top:5px"><i style="width:' + pct + '%"></i></div></td>' +
      '<td class="n">' + j.total_items + '</td>' +
      '<td>' + split + '</td>' +
      '<td class="n">' + ago(j.created_at) + '</td></tr>'
  }).join('')
}

function renderEvents (events) {
  if (!events.length) { $('events').innerHTML = '<div class="muted">Nothing yet.</div>'; return }
  $('events').innerHTML = events.map(e =>
    '<div class="ev"><span class="t">' + esc(new Date(e.server_ts).toLocaleTimeString()) + '</span>' +
    '<b>' + esc(e.type) + '</b>' +
    '<span>' + esc(e.label ?? '') + '</span>' +
    '<span>' + esc(e.detail ?? '') + '</span></div>').join('')
}

function renderKpis (s) {
  $('kpis').innerHTML = [
    ['Computers online', s.online + ' / ' + s.enrolled],
    ['Tasks completed', s.tasksDone],
    ['Jobs run', s.jobs],
    ['Digits classified', s.inferred],
    ['Accuracy', s.accuracy == null ? '—' : s.accuracy + '%'],
  ].map(([l, v]) => '<div class="kpi"><div class="v">' + esc(v) + '</div><div class="l">' + esc(l) + '</div></div>').join('')
}

let failures = 0
async function refresh () {
  try {
    const d = await get('/dashboard/data')
    $('addr').textContent = d.publicOrigin ?? ''
    renderKpis(d.summary); renderCurrent(d.jobs); renderHosts(d.hosts); renderJobs(d.jobs); renderEvents(d.events)
    $('tick').textContent = 'updated ' + new Date().toLocaleTimeString()
    $('tick').classList.remove('err')
    failures = 0
  } catch (err) {
    failures++
    $('tick').textContent = failures > 2 ? 'not responding — is the server running?' : 'retrying…'
    $('tick').classList.add('err')
  }
}
refresh()
setInterval(refresh, 1000)
</script></body></html>`
