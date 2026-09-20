/**
 * Put the orchestrator on the internet with Tailscale Funnel, so a friend can join by
 * pasting one command.
 *
 *   node scripts/share.ts [--port 8080]
 *
 * Why this exists alongside `transport/tailscale` (tsnet).
 *
 * They solve the same problem in opposite directions, and the difference is friction
 * rather than security:
 *
 *   tsnet   every worker embeds a Tailscale node and joins the tailnet. Nothing is
 *           publicly reachable. But each worker needs an auth key, device approval, and
 *           a ~29MB per-platform helper, and each counts against the tailnet's device
 *           limit. Right for a fleet you control.
 *
 *   Funnel  the *server* gets one permanent public HTTPS name. Workers install nothing
 *           and need no Tailscale account -- they speak ordinary WSS. Right for "my
 *           friends are lending me their laptops", which is how every machine on this
 *           fleet actually joined.
 *
 * Funnel is not the weaker option for being public. The device protocol authenticates
 * every connection with a 120-second Ed25519 assertion signed by a key that never leaves
 * the machine, and every result carries its own signature; a stolen URL buys nothing.
 * What Funnel does cost is throughput: traffic crosses a Tailscale relay under an
 * unpublished bandwidth cap. Small task payloads are fine -- measure before shipping
 * large model artifacts through it.
 *
 * Only ports 443, 8443 and 10000 can be funnelled, so this maps the local port onto 443.
 */
import { spawn } from 'node:child_process'
import { createServer } from 'node:net'

const argv = process.argv.slice(2)
const portArg = argv.indexOf('--port')
const PORT = Number(portArg >= 0 ? argv[portArg + 1] : process.env.PORT ?? 8080)

const run = (cmd: string, args: string[]): Promise<{ code: number; out: string }> =>
  new Promise(resolve => {
    const c = spawn(cmd, args)
    let out = ''
    c.stdout.on('data', d => { out += String(d) })
    c.stderr.on('data', d => { out += String(d) })
    c.on('error', () => resolve({ code: 127, out: `${cmd} not found` }))
    c.on('close', code => resolve({ code: code ?? 1, out }))
  })

/** Is something already serving locally? Funnelling a dead port publishes a 502. */
const localPortIsBusy = (port: number): Promise<boolean> =>
  new Promise(resolve => {
    const probe = createServer()
    probe.once('error', () => resolve(true))
    probe.once('listening', () => probe.close(() => resolve(false)))
    probe.listen(port, '127.0.0.1')
  })

const fail = (msg: string): never => { console.error(`\n  ${msg}\n`); process.exit(1) }

if (!(await localPortIsBusy(PORT))) {
  fail(`Nothing is listening on 127.0.0.1:${PORT}.\n` +
       `  Start the orchestrator first, then run this again.`)
}

const ts = await run('tailscale', ['status', '--json'])
if (ts.code === 127) fail('Tailscale is not installed.  brew install --cask tailscale')
if (ts.code !== 0) fail(`Tailscale is not signed in. Run:  tailscale up`)

const dnsName = String(JSON.parse(ts.out).Self?.DNSName ?? '').replace(/\.$/, '')
if (!dnsName) fail('Could not read this machine\'s Tailscale name.')

console.log(`\n  Publishing 127.0.0.1:${PORT} as https://${dnsName}\n`)

const funnel = await run('tailscale', ['funnel', '--bg', String(PORT)])
if (funnel.code !== 0) {
  // The first run on a tailnet prints an enable link instead of failing usefully.
  fail(funnel.out.trim() || 'tailscale funnel failed')
}

const origin = `https://${dnsName}`

// Cloudflare-style tunnels answer before the route exists; Funnel is quicker but not
// instant, and reporting success early is how people end up debugging a working setup.
let reachable = false
for (let i = 0; i < 20 && !reachable; i++) {
  const res = await fetch(`${origin}/healthz`, { signal: AbortSignal.timeout(8000) }).catch(() => null)
  if (res?.ok) reachable = true
  else await new Promise(r => setTimeout(r, 3000))
}

console.log(`  ${'-'.repeat(66)}`)
console.log(`\n  Your network is live at\n\n      ${origin}\n`)
console.log(reachable
  ? '  Verified reachable from the public internet.\n'
  : '  Not answering yet. Funnel may still be provisioning its certificate;\n' +
    '  check again in a minute with:  curl ' + origin + '/healthz\n')
console.log(`  Send a friend this link. It shows them exactly what to run:\n\n      ${origin}/join\n`)
console.log(`  This address is permanent -- it survives restarts, reboots and network changes.`)
console.log(`  Stop sharing with:  tailscale funnel --https=443 off`)
console.log(`  ${'-'.repeat(66)}\n`)
