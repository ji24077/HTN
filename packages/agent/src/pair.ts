import { hostname } from 'node:os'
import { diagnoseOrigin } from '@dwp/protocol'
import { installDnsFallback, dnsFallbackEnabled } from './resolver.ts'
import { ensureKeypair } from './keys.ts'
import { saveConfig } from './config.ts'

export async function pair(server: string, code: string, label?: string): Promise<void> {
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
    const hint = diagnosis.cause === 'dns-local-only' && !dnsFallbackEnabled()
      ? `\n  Or retry with the built-in workaround:\n` +
        `      DWP_DNS_FALLBACK=1 pnpm agent pair --server ${origin} --code ${code}\n`
      : ''
    console.error(`\nCould not reach the server.\n\n  ${diagnosis.message}\n${hint}`)
    process.exit(1)
  }

  if (!res.ok) {
    const body = await res.text().catch(() => '')
    if (res.status === 400) {
      console.error(`\nThat pairing code was not accepted.\n\n` +
        `  Codes expire ten minutes after they are created and work only once.\n` +
        `  Ask for a fresh link.\n`)
    } else if (res.status === 429) {
      console.error(`\nToo many attempts. Wait a few minutes and try again.\n`)
    } else {
      console.error(`\nPairing failed (HTTP ${res.status}). ${body.slice(0, 200)}\n`)
    }
    process.exit(1)
  }

  const { hostId, label: assigned, wsUrl } = await res.json() as { hostId: string; label: string; wsUrl: string }
  saveConfig({
    server: origin, wsUrl, hostId, label: assigned,
    allowCompute: true, allowBrowser: false, maxConcurrency: 2,
  })
  console.log(`Paired as "${assigned}"\n  host id : ${hostId}\n  control : ${origin}`)
  console.log(`\nStart it with:  pnpm agent run`)
}
