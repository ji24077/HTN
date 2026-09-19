/**
 * Watch the stickman learn.
 *
 * It imports the same physics file the agents ran, so what is animated is literally the
 * gait that was scored — not a second implementation that might disagree.
 */
export const walkerHtml = (): string => `<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Walker</title>
<style>
  :root{--bg:#eef1f3;--card:#fff;--ink:#10161b;--ink2:#51616b;--ink3:#7d8992;--rule:#dde3e7;
    --accent:#0b6e8c;--ok:#2c6b45;--ground:#c7d0d6;--fig:#10161b}
  @media (prefers-color-scheme:dark){:root{--bg:#0d1215;--card:#161c21;--ink:#e4eaed;--ink2:#a9b6bf;
    --ink3:#78858e;--rule:#28333a;--accent:#4bb2d1;--ok:#6cbb8c;--ground:#2b353c;--fig:#e4eaed}}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);padding:20px 16px 50px;
    font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
  main{max-width:1000px;margin:0 auto}
  header{display:flex;flex-wrap:wrap;gap:8px 16px;align-items:baseline;margin-bottom:16px}
  h1{font-size:19px;margin:0}
  .sub{color:var(--ink3);font-size:13px}
  .live{margin-left:auto;font-size:12px;color:var(--ink3)}
  .stage{background:var(--card);border:1px solid var(--rule);border-radius:10px;overflow:hidden}
  canvas{display:block;width:100%;height:auto;background:transparent}
  .kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px;margin:14px 0}
  .kpi{background:var(--card);border:1px solid var(--rule);border-radius:8px;padding:11px 13px}
  .kpi .v{font:600 20px/1.2 ui-monospace,SFMono-Regular,Menlo,monospace;letter-spacing:-.02em}
  .kpi .l{font-size:10.5px;letter-spacing:.07em;text-transform:uppercase;color:var(--ink3);margin-top:3px}
  h2{font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--ink3);margin:24px 0 8px}
  .chart{background:var(--card);border:1px solid var(--rule);border-radius:10px;padding:12px}
  .hosts{display:flex;flex-wrap:wrap;gap:8px}
  .host{background:var(--card);border:1px solid var(--rule);border-radius:99px;padding:5px 13px;font-size:13px}
  .host b{font-family:ui-monospace,monospace}
  .wait{padding:50px 20px;text-align:center;color:var(--ink3)}
  code{background:var(--bg);padding:2px 6px;border-radius:4px;font-size:13px}
</style></head><body><main>

<header>
  <h1>Learning to walk</h1>
  <span class="sub">evolved across every computer on the network</span>
  <span class="live" id="live">connecting…</span>
</header>

<div class="stage"><canvas id="c" width="1000" height="360"></canvas></div>
<div class="kpis" id="kpis"></div>

<h2>Best distance per generation</h2>
<div class="chart"><canvas id="chart" width="1000" height="180"></canvas></div>

<h2>Who evaluated this generation</h2>
<div class="hosts" id="hosts"></div>

</main>
<script type="module">
import { trace } from '/walker.js'

const c = document.getElementById('c'), ctx = c.getContext('2d')
const chart = document.getElementById('chart'), cctx = chart.getContext('2d')
const css = getComputedStyle(document.documentElement)
const col = n => css.getPropertyValue(n).trim()

let frames = [], frame = 0, state = null, lastGen = -1

function drawWaiting (msg) {
  ctx.clearRect(0, 0, c.width, c.height)
  ctx.fillStyle = col('--ink3'); ctx.font = '15px ui-sans-serif,system-ui'; ctx.textAlign = 'center'
  ctx.fillText(msg, c.width / 2, c.height / 2)
}

function draw () {
  ctx.clearRect(0, 0, c.width, c.height)
  if (!frames.length) { drawWaiting('waiting for the first generation…'); return }

  const f = frames[frame % frames.length]
  const scale = 150
  const groundY = c.height - 70
  // Track the figure so it stays centred as it travels.
  const camX = c.width / 2 - f.x * scale

  // Ground, with markers every half metre so motion is obvious.
  ctx.strokeStyle = col('--ground'); ctx.lineWidth = 2
  ctx.beginPath(); ctx.moveTo(0, groundY); ctx.lineTo(c.width, groundY); ctx.stroke()
  ctx.fillStyle = col('--ground')
  for (let m = -4; m < 40; m += 0.5) {
    const x = camX + m * scale
    if (x < -10 || x > c.width + 10) continue
    ctx.fillRect(x, groundY + 3, m % 1 === 0 ? 2 : 1, m % 1 === 0 ? 10 : 5)
    if (m % 1 === 0 && m >= 0) {
      ctx.font = '10px ui-monospace,monospace'; ctx.textAlign = 'center'
      ctx.fillText(m + 'm', x, groundY + 26)
    }
  }

  const px = (wx, wy) => [camX + wx * scale, groundY - wy * scale]
  ctx.strokeStyle = col('--fig'); ctx.lineWidth = 4
  ctx.lineCap = 'round'; ctx.lineJoin = 'round'

  // Torso, from hips up.
  const [hx, hy] = px(f.x, f.y)
  const headX = hx + Math.sin(f.th) * 52, headY = hy - Math.cos(f.th) * 52
  ctx.beginPath(); ctx.moveTo(hx, hy); ctx.lineTo(headX, headY); ctx.stroke()
  ctx.beginPath(); ctx.arc(headX, headY - 13, 13, 0, Math.PI * 2)
  ctx.fillStyle = col('--fig'); ctx.fill()

  // Arms, swung opposite the legs — cosmetic, but it reads as walking.
  const swing = (f.l[4] - f.r[4]) * 0.8
  ctx.lineWidth = 3
  for (const s of [1, -1]) {
    const sx = hx + Math.sin(f.th) * 34, sy = hy - Math.cos(f.th) * 34
    ctx.beginPath(); ctx.moveTo(sx, sy)
    ctx.lineTo(sx + s * swing * scale * 0.5, sy + 34)
    ctx.stroke()
  }

  // Legs, and the feet that made standing possible at all.
  ctx.lineWidth = 4
  for (const [leg, alpha] of [[f.l, 1], [f.r, 0.55]]) {
    ctx.globalAlpha = alpha
    const [ax, ay] = px(leg[0], leg[1])
    const [bx, by] = px(leg[2], leg[3])
    const [dx, dy] = px(leg[4], leg[5])
    ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.lineTo(dx, dy); ctx.stroke()
    const [hxp] = px(leg[6], leg[5])
    const [txp] = px(leg[7], leg[5])
    ctx.lineWidth = 5
    ctx.beginPath(); ctx.moveTo(hxp, dy); ctx.lineTo(txp, dy); ctx.stroke()
    ctx.lineWidth = 4
  }
  ctx.globalAlpha = 1

  frame++
}

function drawChart (history) {
  cctx.clearRect(0, 0, chart.width, chart.height)
  if (!history || history.length < 2) return
  const w = chart.width, h = chart.height, pad = 26
  const max = Math.max(...history.map(p => p.distance), 1)
  const n = history.length

  cctx.strokeStyle = col('--rule'); cctx.lineWidth = 1
  cctx.beginPath(); cctx.moveTo(pad, h - pad); cctx.lineTo(w - 8, h - pad); cctx.stroke()

  cctx.strokeStyle = col('--accent'); cctx.lineWidth = 2; cctx.beginPath()
  history.forEach((p, i) => {
    const x = pad + (i / (n - 1)) * (w - pad - 12)
    const y = (h - pad) - (p.distance / max) * (h - pad - 12)
    i === 0 ? cctx.moveTo(x, y) : cctx.lineTo(x, y)
  })
  cctx.stroke()

  cctx.fillStyle = col('--ink3'); cctx.font = '10px ui-monospace,monospace'; cctx.textAlign = 'left'
  cctx.fillText(max.toFixed(1) + 'm', 2, 12)
  cctx.fillText('gen ' + history[0].gen, pad, h - 8)
  cctx.textAlign = 'right'
  cctx.fillText('gen ' + history[n - 1].gen, w - 8, h - 8)
}

async function poll () {
  try {
    const res = await fetch('/experiments/walker', { credentials: 'same-origin' })
    if (res.status === 404) {
      document.getElementById('live').textContent = 'no run yet'
      drawWaiting('no run started yet')
      return
    }
    const { state: s } = await res.json()
    state = s
    document.getElementById('live').textContent =
      'generation ' + s.generation + ' of ' + s.totalGenerations

    if (s.generation !== lastGen && Array.isArray(s.genome)) {
      lastGen = s.generation
      // Replay the current champion, using the same physics that scored it.
      frames = trace(s.genome, Math.min(s.steps ?? 1200, 1400))
      frame = 0
    }

    document.getElementById('kpis').innerHTML = [
      ['Generation', s.generation + ' / ' + s.totalGenerations],
      ['Distance walked', (s.best?.distance ?? 0).toFixed(2) + ' m'],
      ['Fitness', (s.best?.fitness ?? 0).toFixed(2)],
      ['Upright', Math.round(100 * (s.best?.ticks ?? 0) / (s.steps || 1)) + '%'],
      ['Gaits tested', (s.evaluations ?? 0).toLocaleString()],
      ['Elapsed', (s.elapsedSeconds ?? 0) + 's'],
    ].map(([l, v]) => '<div class="kpi"><div class="v">' + v + '</div><div class="l">' + l + '</div></div>').join('')

    document.getElementById('hosts').innerHTML =
      Object.entries(s.hosts ?? {}).map(([k, v]) =>
        '<div class="host">' + k.replace(/[<>&]/g, '') + ' <b>' + v + '</b> gaits</div>').join('')
        || '<div class="host">waiting…</div>'

    drawChart(s.history)
  } catch {
    document.getElementById('live').textContent = 'not responding'
  }
}

drawWaiting('waiting for the first generation…')
poll(); setInterval(poll, 1500)
setInterval(draw, 1000 / 40)
</script></body></html>`
