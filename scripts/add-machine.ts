/**
 * Everything needed to put one more machine on the network, printed as one command.
 *
 *   node scripts/add-machine.ts
 *   node scripts/add-machine.ts --label ethans-desktop --count 3
 *
 * Adding a machine needs three things to line up: an address that machine can actually
 * reach, an invite that has not expired, and an image it can pull. Each is easy and the
 * combination is where every attempt so far has gone wrong — most often by handing
 * someone a `127.0.0.1` URL, which is a perfectly valid address for a completely
 * different computer.
 *
 * So this checks the address is reachable from somewhere other than here, mints the
 * invite, and prints the line to paste. It does not publish anything or change any
 * settings.
 */
import { execFile } from 'node:child_process'
import { readFileSync } from 'node:fs'
import { networkInterfaces } from 'node:os'
import { promisify } from 'node:util'
import { join } from 'node:path'
import { homedir } from 'node:os'

const exec = promisify(execFile)

const argv = process.argv.slice(2)
const opt = (name: string, fallback: string): string => {
  const i = argv.indexOf(`--${name}`)
  return i >= 0 && argv[i + 1] !== undefined ? argv[i + 1]! : fallback
}

const LOCAL = opt('server', process.env.DWP_CONTROL ?? 'http://127.0.0.1:8080')
const COUNT = Math.max(1, Number(opt('count', '1')))
const LABEL = opt('label', '')
/**
 * Which host port the window is published on.
 *
 * Not always 43117: anything already running the desktop agent holds that, and the
 * container then fails to start with "address already in use". Changing only the host
 * half is the fix, and the agent is told about it so the URL it prints is the one that
 * actually works.
 */
const GUI_PORT = opt('gui-port', '43117')
const IMAGE = opt('image', process.env.DWP_AGENT_IMAGE ?? 'ghcr.io/notjackl3/dwp-agent:latest')
const TOKEN_FILE = process.env.DWP_ADMIN_TOKEN_FILE ?? join(homedir(), '.dwp', 'admin-token')

const die = (msg: string): never => { console.error(`\n  ${msg}\n`); process.exit(1) }

function adminToken(): string {
  try {
    return readFileSync(TOKEN_FILE, 'utf8').trim()
  } catch {
    return die(`No admin token at ${TOKEN_FILE}.\n  Start the control service once with ./scripts/start-fleet.sh — it creates one.`)
  }
}

/** Every address of this machine that is not loopback, newest-looking first. */
function lanAddresses(): string[] {
  const found: string[] = []
  for (const list of Object.values(networkInterfaces())) {
    for (const nic of list ?? []) {
      if (nic.family === 'IPv4' && !nic.internal) found.push(nic.address)
    }
  }
  return found
}

/** This machine's public Tailscale Funnel name, if it is serving one. */
async function funnelOrigin(): Promise<string | null> {
  const status = await exec('tailscale', ['status', '--json']).catch(() => null)
  if (!status) return null
  let name: string
  try {
    name = String((JSON.parse(status.stdout) as { Self?: { DNSName?: string } }).Self?.DNSName ?? '')
      .replace(/\.$/, '')
  } catch { return null }
  if (!name) return null
  // Serving is not the same as installed. Ask whether anything is actually funnelled.
  const funnel = await exec('tailscale', ['funnel', 'status']).catch(() => null)
  if (!funnel || /no serve config/i.test(funnel.stdout + funnel.stderr)) return null
  return `https://${name}`
}

/** Does this address answer, and is it one another machine could use? */
async function reachable(origin: string): Promise<boolean> {
  const res = await fetch(`${origin}/healthz`, { signal: AbortSignal.timeout(5_000) }).catch(() => null)
  return res?.ok === true
}

async function invite(origin: string, token: string): Promise<string> {
  const res = await fetch(`${origin}/v1/device-invites`, {
    method: 'POST',
    headers: { authorization: `Bearer ${token}` },
  })
  if (res.status === 429) {
    return die('The control service has issued its limit of invites for this hour (ten).\n'
      + '  Wait, or use the ones you already created — each works once, for ten minutes.')
  }
  if (!res.ok) return die(`Could not create an invite: HTTP ${res.status} ${await res.text()}`)
  return (await res.json() as { code: string }).code
}

async function main(): Promise<void> {
  const token = adminToken()

  if (!await reachable(LOCAL)) {
    die(`Nothing is answering at ${LOCAL}.\n  Start it with:  ./scripts/start-fleet.sh`)
  }

  /**
   * Which address to hand out, in order of how far it reaches.
   *
   * A Funnel name works from anywhere on the internet, which is the case this project is
   * actually for — other people's laptops, in other places. A LAN address works for
   * machines in the same building. Loopback works for nothing but this computer, and
   * handing it out is the single most common way this goes wrong: the command looks
   * right, and the other machine spends two minutes failing to reach itself.
   */
  const funnel = await funnelOrigin()
  const lan = lanAddresses()
  const port = new URL(LOCAL).port || '80'

  let origin: string | null = null
  let scope = ''
  if (funnel && await reachable(funnel)) {
    origin = funnel
    scope = 'anywhere on the internet'
  } else {
    for (const address of lan) {
      const candidate = `http://${address}:${port}`
      if (await reachable(candidate)) { origin = candidate; scope = 'machines on this network'; break }
    }
  }

  if (!origin) {
    console.error(`\n  ${LOCAL} answers here, but no address another machine could use does.\n`)
    console.error(`  The control service is probably bound to 127.0.0.1. Other machines cannot reach that.\n`)
    console.error(`  For machines on this network, restart it listening on all interfaces:\n`)
    console.error(`      LISTEN_HOST=0.0.0.0 ./scripts/start-fleet.sh --replace\n`)
    console.error(`  Then it will be at  http://${lan[0] ?? '<this machine’s IP>'}:${port}\n`)
    console.error(`  For machines anywhere else, publish it over Tailscale Funnel:\n`)
    console.error(`      node scripts/share.ts --port ${port}\n`)
    process.exit(1)
  }

  console.log(`\n  Control service  ${origin}`)
  console.log(`  Reachable from   ${scope}`)
  console.log(`  Image            ${IMAGE}\n`)

  for (let i = 1; i <= COUNT; i += 1) {
    const code = await invite(origin, token)
    const label = LABEL === '' ? '' : (COUNT === 1 ? LABEL : `${LABEL}-${i}`)
    console.log(`  ${'─'.repeat(72)}`)
    console.log(`  Machine ${i} of ${COUNT}${label ? ` — ${label}` : ''}. Paste this on that computer:\n`)
    // One line. Backslash continuations are POSIX-only and break in PowerShell, which is
    // where a Windows machine's owner will paste this.
    const name = COUNT === 1 ? 'dwp-agent' : `dwp-agent-${i}`
    const guiPort = Number(GUI_PORT) + (i - 1)
    console.log(`      docker run -d --name ${name} --restart unless-stopped`
      + ` -v ${name}-data:/data -p 127.0.0.1:${guiPort}:43117`
      + ` -e DWP_INVITE="${origin}/join?code=${code}"`
      + (label ? ` -e DWP_LABEL="${label}"` : '')
      + ` -e DWP_GUI_PUBLIC_ORIGIN="http://127.0.0.1:${guiPort}"`
      + ` -e DWP_IMAGE="${IMAGE}" ${IMAGE}\n`)
    console.log(`      window on that machine:  http://127.0.0.1:${guiPort}/`)
    console.log(`      if that port is taken:   add --gui-port <free port> and run this again\n`)
  }
  console.log(`  ${'─'.repeat(72)}\n`)
  console.log(`  Each invite works once and expires in ten minutes.`)
  console.log(`  The window needs its path token, which that machine prints:  docker logs dwp-agent`)
  console.log(`  Watch them arrive here:  ${LOCAL}\n`)
}

main().catch((err: unknown) => die(err instanceof Error ? err.message : String(err)))
