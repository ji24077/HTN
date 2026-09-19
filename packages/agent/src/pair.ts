import { hostname } from 'node:os'
import { diagnoseOrigin } from '@dwp/protocol'
import { installDnsFallback, dnsFallbackEnabled } from './resolver.ts'
import { invocation } from './paths.ts'
import { ensureKeypair } from './keys.ts'
import { saveConfig } from './config.ts'
import { rememberTelemetrySecret } from './telemetry.ts'

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

  const { hostId, label: assigned, wsUrl, releaseKey, releaseVersion } = await res.json() as {
    hostId: string; label: string; wsUrl: string
    releaseKey?: string | null; releaseVersion?: string | null
  }
  saveConfig({
    server: origin, wsUrl, hostId, label: assigned,
    allowCompute: true, allowBrowser: false, maxConcurrency: 2,
    // Pinned once, here. Every future update is checked against this and nothing else.
    releaseKey: releaseKey ?? null,
    installedRelease: releaseVersion ?? null,
    autoUpdate: true,
  })
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
  console.log(`\nStart it with:  ${invocation()} run`)
}
