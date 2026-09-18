import { hostname } from 'node:os'
import { ensureKeypair } from './keys.ts'
import { saveConfig } from './config.ts'

export async function pair(server: string, code: string, label?: string): Promise<void> {
  const { publicKeySpki } = ensureKeypair()
  const origin = server.replace(/\/+$/, '')

  const res = await fetch(`${origin}/hosts/pair`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ code: code.trim().toUpperCase(), publicKey: publicKeySpki, label: label ?? hostname() }),
  })
  if (!res.ok) {
    const body = await res.text()
    throw new Error(`pairing rejected (${res.status}): ${body}`)
  }

  const { hostId, label: assigned, wsUrl } = await res.json() as { hostId: string; label: string; wsUrl: string }
  saveConfig({
    server: origin, wsUrl, hostId, label: assigned,
    allowCompute: true, allowBrowser: false, maxConcurrency: 2,
  })
  console.log(`Paired as "${assigned}"\n  host id : ${hostId}\n  control : ${origin}`)
  console.log(`\nStart it with:  pnpm agent run`)
}
