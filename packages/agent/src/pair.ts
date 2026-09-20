import { readFileSync } from 'node:fs'
import { hostname } from 'node:os'
import { setTimeout as sleep } from 'node:timers/promises'
import { createLogger, diagnoseOrigin, WorkerTelemetry } from '@dwp/protocol'
import { installDnsFallback, dnsFallbackEnabled } from './resolver.ts'
import { invocation } from './paths.ts'
import { ensureKeypair } from './keys.ts'
import { loadConfig, saveConfig, type AgentConfig } from './config.ts'
import { initTelemetry, rememberTelemetrySecret } from './telemetry.ts'

const log = createLogger({ component: 'agent' })

export type PairOutcome =
  | { ok: true; hostId: string; label: string; server: string; pinnedKey: boolean }
  | { ok: false; message: string; hint?: string }

/**
 * Enroll this computer, and *return* what happened.
 *
 * Split from the CLI wrapper below because the desktop window needs the failure as a
 * string it can render. The original printed to the console and called `process.exit`,
 * which from a window would have killed the app with the explanation going nowhere
 * anybody could read it.
 */
export async function pairHost(server: string, code: string, label?: string): Promise<PairOutcome> {
  rememberTelemetrySecret(code.trim())
  rememberTelemetrySecret(code.trim().toUpperCase())
  installDnsFallback()
  const { publicKeySpki } = ensureKeypair()
  const origin = server.replace(/\/+$/, '')

  let res: Response
  try {
    res = await fetch(`${origin}/hosts/pair`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ code: code.trim().toUpperCase(), publicKey: publicKeySpki, label: label ?? hostname() }),
    })
  } catch {
    // A raw "fetch failed" tells the person nothing. Work out the actual cause and say
    // what they can do about it — this is the first thing anyone joining ever sees go
    // wrong, and usually the only error they will ever read.
    const diagnosis = await diagnoseOrigin(origin)
    // PowerShell and cmd have no inline `KEY=value command` form, so the suggestion has
    // to be written the way the machine reading it can actually run.
    const retry = process.platform === 'win32'
      ? `$env:DWP_DNS_FALLBACK='1'; pnpm agent pair --server ${origin} --code ${code}`
      : `DWP_DNS_FALLBACK=1 pnpm agent pair --server ${origin} --code ${code}`
    return {
      ok: false,
      message: `Could not reach the server. ${diagnosis.message}`,
      hint: diagnosis.cause === 'dns-local-only' && !dnsFallbackEnabled()
        ? `Or retry with the built-in workaround:\n    ${retry}`
        : undefined,
    }
  }

  if (!res.ok) {
    const body = await res.text().catch(() => '')
    if (res.status === 400) {
      return {
        ok: false,
        message: 'That invite was not accepted.',
        hint: 'Invites expire ten minutes after they are created and work only once. Ask for a fresh one.',
      }
    }
    if (res.status === 429) return { ok: false, message: 'Too many attempts. Wait a few minutes and try again.' }
    return { ok: false, message: `Pairing failed (HTTP ${res.status}). ${body.slice(0, 200)}` }
  }

  const { hostId, label: assigned, wsUrl, releaseKey, releaseVersion, telemetry } = await res.json() as {
    hostId: string; label: string; wsUrl: string
    releaseKey?: string | null; releaseVersion?: string | null
    telemetry?: unknown
  }
  const config: AgentConfig = {
    server: origin, wsUrl, hostId, label: assigned,
    allowCompute: true, allowBrowser: false, maxConcurrency: 2,
    // Pinned once, here. Every future update is checked against this and nothing else.
    releaseKey: releaseKey ?? null,
    installedRelease: releaseVersion ?? null,
    autoUpdate: true,
  }
  const parsed = WorkerTelemetry.safeParse(telemetry)
  if (parsed.success) config.telemetry = parsed.data
  saveConfig(config)
  initTelemetry(config.telemetry)
  return { ok: true, hostId, label: assigned, server: origin, pinnedKey: Boolean(releaseKey) }
}

export async function pair(server: string, code: string, label?: string): Promise<void> {
  const result = await pairHost(server, code, label)
  if (!result.ok) {
    console.error(`\n${result.message}\n${result.hint ? `\n  ${result.hint}\n` : ''}`)
    process.exit(1)
  }
  console.log(`Paired as "${result.label}"\n  host id : ${result.hostId}\n  control : ${result.server}`)
  console.log(result.pinnedKey
    ? `  updates : signed releases, key pinned`
    : `  updates : this server offers no signed releases, so updates stay manual`)
  /**
   * Offer the durable option first, and the terminal one second.
   *
   * This line used to say only "Start it with: <exe> run", and that is how a Windows
   * laptop came to be joined to the network by a PowerShell window — which its owner
   * then closed. The agent's whole lifetime was that window's, and nothing anywhere in
   * the flow mentioned that `install-service` existed. Whichever line someone copies,
   * they should be told there is a version of this that survives a restart.
   */
  console.log(`\nStart it now:            ${invocation()} run`)
  console.log(`Or have it start itself: ${invocation()} install-service`)
  console.log(`\n  install-service rejoins after every restart, with no window left open.`)
  console.log(`  Without it, this computer is on the network only while that command is running.\n`)
}

// ------------------------------------------------------- joining without a person

/**
 * Accept either a full invite link or a bare code, and say which field is missing.
 *
 * Lives here rather than in the window because a container joins from an environment
 * variable holding exactly the same string a person would paste, and two parsers for one
 * format is one parser too many — the window's copy already disagreed with this path
 * about whether a bare code was usable.
 */
export function parseInvite(
  text: string,
  fallbackServer?: string,
): { server: string; code: string } | { error: string } {
  const trimmed = text.trim()
  if (trimmed === '') return { error: 'Paste the invite link you were sent.' }

  if (/^https?:\/\//i.test(trimmed)) {
    let url: URL
    try { url = new URL(trimmed) } catch {
      return { error: 'That does not look like a link. Paste the whole thing, starting with https://' }
    }
    const code = url.searchParams.get('code')
    if (!code) return { error: 'That link has no invite code in it. It should end with ?code=SOMETHING' }
    return { server: url.origin, code }
  }

  // A bare code is only usable if we already know where to send it.
  if (!fallbackServer) {
    return {
      error: 'That looks like just the code. Paste the whole invite link instead, '
        + 'so this computer knows which network to join.',
    }
  }
  return { server: fallbackServer.replace(/\/+$/, ''), code: trimmed }
}

export type EnrolmentSource = { invite: string; from: string; label?: string }

/**
 * The invite this process was started with, if it was started with one.
 *
 * A container has nobody to paste a link into a window, so the link arrives the way
 * everything else arrives in a container: in the environment. The file form exists
 * because an invite is a credential, and a Docker or Compose secret is a file — putting
 * it in `environment:` writes it into `docker inspect` and every log that dumps env.
 */
export function enrolmentFromEnvironment(): EnrolmentSource | null {
  const label = process.env.DWP_LABEL?.trim() || undefined

  const file = process.env.DWP_INVITE_FILE
  if (file) {
    try {
      const text = readFileSync(file, 'utf8').trim()
      if (text !== '') return { invite: text, from: `DWP_INVITE_FILE (${file})`, label }
    } catch (err) {
      log.warn('enrol.invite_file_unreadable', { path: file, err: String(err) })
    }
  }

  const invite = process.env.DWP_INVITE?.trim()
  if (invite) return { invite, from: 'DWP_INVITE', label }

  const server = process.env.DWP_SERVER?.trim()
  const code = process.env.DWP_CODE?.trim()
  if (server && code) return { invite: `${server.replace(/\/+$/, '')}/join?code=${code}`, from: 'DWP_SERVER/DWP_CODE', label }

  return null
}

/**
 * Join from the environment, waiting for a control service that may still be starting.
 *
 * Compose starts everything at once, so the first few attempts failing is the ordinary
 * case rather than an error — `depends_on` can only wait for a container, not for a
 * server inside it that is still opening its database. Retrying with a ceiling turns
 * that into a slower start instead of a container that exits and takes its invite with
 * it.
 *
 * An invite is single-use and expires, so this must not burn one on a machine that has
 * already joined: an agent with a config ignores the variable entirely, which is what
 * lets the same compose file be brought up twice without re-enrolling.
 */
export async function autoEnrol(opts: { attempts?: number } = {}): Promise<PairOutcome | null> {
  if (loadConfig()) return null
  const source = enrolmentFromEnvironment()
  if (!source) return null

  const parsed = parseInvite(source.invite, process.env.DWP_SERVER?.trim())
  if ('error' in parsed) {
    log.error('enrol.invite_unusable', { from: source.from, detail: parsed.error })
    console.error(`\n  The invite in ${source.from} is not usable: ${parsed.error}\n`)
    return { ok: false, message: parsed.error }
  }

  const attempts = opts.attempts ?? 30
  let last: PairOutcome = { ok: false, message: 'never attempted' }
  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    last = await pairHost(parsed.server, parsed.code, source.label)
    if (last.ok) {
      log.info('enrol.joined', { hostId: last.hostId, label: last.label, server: last.server, from: source.from })
      console.log(`\n  Joined ${last.server} as "${last.label}" (${last.hostId}), from ${source.from}.\n`)
      return last
    }
    /**
     * Stop on a refusal, keep going on an unreachable server.
     *
     * A rejected invite is rejected for good — it has been used, or it has expired — and
     * retrying it thirty times only spends the pairing rate limit that protects the
     * server. Not being able to reach the server yet is the opposite: it is expected,
     * and it fixes itself.
     */
    if (!/could not reach/i.test(last.message)) {
      log.error('enrol.refused', { from: source.from, detail: last.message })
      console.error(`\n  Could not join: ${last.message}\n`)
      return last
    }
    if (attempt < attempts) {
      log.info('enrol.waiting_for_server', { attempt, attempts, server: parsed.server })
      await sleep(Math.min(1_000 * attempt, 5_000))
    }
  }
  console.error(`\n  Gave up joining ${parsed.server} after ${attempts} attempts: ${last.message}\n`)
  return last
}
