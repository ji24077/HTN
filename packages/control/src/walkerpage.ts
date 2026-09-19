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
  .hosts{display:flex;flex-direction:column;gap:9px}
  .host{background:var(--card);border:1px solid var(--rule);border-radius:9px;padding:9px 12px;font-size:13px}
  .host b{font-family:ui-monospace,monospace}
  .hrow{display:flex;align-items:baseline;justify-content:space-between;gap:10px;margin-bottom:6px}
  .hname{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .hos{font-size:11px;opacity:.65;margin-left:6px;text-transform:uppercase;letter-spacing:.04em}
  .hbar{height:7px;border-radius:99px;background:var(--rule);overflow:hidden}
  .hbar i{display:block;height:100%;border-radius:99px;background:currentColor}
  .h-darwin{color:#5b9dff}.h-win32{color:#46c08a}.h-ios{color:#c07ae0}.h-other{color:#9aa4b2}
  .legend{display:flex;flex-wrap:wrap;gap:14px;margin-top:10px}
  .lg{display:flex;align-items:center;gap:7px;font-size:12px;color:var(--ink2)}
  .sw{width:11px;height:11px;border-radius:3px;flex:none}
  .lg b{font-family:ui-monospace,monospace;color:var(--ink)}
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
<p class="sub">Share of the population each machine scored in the latest generation.</p>
<div class="hosts" id="hosts"></div>

<h2>Contribution over time</h2>
<p class="sub">Tasks completed per machine, accumulating across every run — not just this one.
  <span id="contribWindow"></span></p>
<div class="chart"><canvas id="contrib" width="1000" height="220"></canvas></div>
<div class="legend" id="contribLegend"></div>

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

    const split = Object.entries(s.hosts ?? {}).sort((a, b) => b[1] - a[1])
    const total = split.reduce((n, [, v]) => n + v, 0)
    document.getElementById('hosts').innerHTML = split.map(([k, v]) => {
      // The OS is not in this payload, so infer it from the label the host chose.
      // Wrong guesses only mistint a bar; the counts stay whatever the server said.
      const os = /iphone|ipad/i.test(k) ? 'ios'
        : /laptop-|desktop-|win/i.test(k) ? 'win32'
        : /mac|darwin|\.local|campus|eduroam/i.test(k) ? 'darwin' : 'other'
      const pct = total ? (100 * v / total) : 0
      return '<div class="host h-' + os + '">' +
        '<div class="hrow"><span class="hname">' + k.replace(/[<>&]/g, '') +
        '<span class="hos">' + os.replace('win32', 'windows').replace('darwin', 'macos') + '</span></span>' +
        '<span><b>' + v + '</b> gaits · ' + pct.toFixed(0) + '%</span></div>' +
        '<div class="hbar"><i style="width:' + pct.toFixed(1) + '%"></i></div></div>'
    }).join('') || '<div class="host">waiting…</div>'

    drawChart(s.history)
  } catch {
    document.getElementById('live').textContent = 'not responding'
  }
}

// ---------------------------------------------------------- contribution chart
const PALETTE = ['#5b9dff', '#46c08a', '#c07ae0', '#e0a23a', '#e06a6a', '#3ac0c0', '#9aa4b2']
const ccv = document.getElementById('contrib')
const ctx2 = ccv.getContext('2d')

function drawContribution (data) {
  const w = ccv.width, h = ccv.height, padL = 46, padR = 12, padT = 12, padB = 24
  ctx2.clearRect(0, 0, w, h)
  const css = getComputedStyle(document.body)
  const ink3 = css.getPropertyValue('--ink3') || '#8a93a0'
  const rule = css.getPropertyValue('--rule') || '#2a2f3a'

  const series = data?.series ?? []
  const n = data?.buckets?.length ?? 0
  if (!series.length || !n) {
    ctx2.fillStyle = ink3; ctx2.font = '13px system-ui'; ctx2.textAlign = 'center'
    ctx2.fillText('no completed work in this window yet', w / 2, h / 2)
    return
  }

  // Accumulate each series, then stack them. Cumulative is the point: a machine that
  // stops working should flatten, not vanish.
  const cum = series.map(s => { let t = 0; return s.points.map(v => (t += v)) })
  const stack = []
  for (let i = 0; i < cum.length; i++) {
    stack.push(cum[i].map((v, j) => v + (i ? stack[i - 1][j] : 0)))
  }
  const top = stack[stack.length - 1]
  const max = Math.max(1, ...top)

  const x = i => padL + (w - padL - padR) * (n === 1 ? 0.5 : i / (n - 1))
  const y = v => h - padB - (h - padT - padB) * (v / max)

  // Horizontal guides, labelled with real totals.
  ctx2.strokeStyle = rule; ctx2.fillStyle = ink3
  ctx2.font = '11px ui-monospace,monospace'; ctx2.textAlign = 'right'; ctx2.lineWidth = 1
  for (let g = 0; g <= 2; g++) {
    const v = Math.round((max / 2) * g), yy = Math.round(y(v)) + 0.5
    ctx2.beginPath(); ctx2.moveTo(padL, yy); ctx2.lineTo(w - padR, yy); ctx2.stroke()
    ctx2.fillText(v.toLocaleString(), padL - 6, yy + 4)
  }

  // Paint top-down so each band sits over the one beneath it.
  for (let i = stack.length - 1; i >= 0; i--) {
    ctx2.beginPath()
    ctx2.moveTo(x(0), y(0))
    for (let j = 0; j < n; j++) ctx2.lineTo(x(j), y(stack[i][j]))
    ctx2.lineTo(x(n - 1), y(0))
    ctx2.closePath()
    ctx2.fillStyle = PALETTE[i % PALETTE.length] + '55'
    ctx2.fill()
    ctx2.beginPath()
    for (let j = 0; j < n; j++) j ? ctx2.lineTo(x(j), y(stack[i][j])) : ctx2.moveTo(x(j), y(stack[i][j]))
    ctx2.strokeStyle = PALETTE[i % PALETTE.length]; ctx2.lineWidth = 1.5; ctx2.stroke()
  }

  const first = new Date(data.buckets[0]), last = new Date(data.buckets[n - 1])
  const hhmm = d => String(d.getHours()).padStart(2, '0') + ':' + String(d.getMinutes()).padStart(2, '0')
  ctx2.fillStyle = ink3; ctx2.font = '11px system-ui'
  ctx2.textAlign = 'left'; ctx2.fillText(hhmm(first), padL, h - 7)
  ctx2.textAlign = 'right'; ctx2.fillText(hhmm(last), w - padR, h - 7)

  document.getElementById('contribLegend').innerHTML = series.map((s, i) => {
    const total = cum[i][n - 1]
    return '<span class="lg"><span class="sw" style="background:' + PALETTE[i % PALETTE.length] +
      '"></span>' + s.label.replace(/[<>&]/g, '').slice(0, 34) + ' <b>' + total.toLocaleString() + '</b></span>'
  }).join('')
}

async function pollContribution () {
  try {
    const res = await fetch('/contribution?minutes=180', { credentials: 'same-origin' })
    if (!res.ok) return
    drawContribution(await res.json())
    document.getElementById('contribWindow').textContent = 'Last 3 hours.'
  } catch { /* leave the last good chart on screen */ }
}

drawWaiting('waiting for the first generation…')
poll(); setInterval(poll, 1500)
pollContribution(); setInterval(pollContribution, 5000)
setInterval(draw, 1000 / 40)
</script></body></html>`
