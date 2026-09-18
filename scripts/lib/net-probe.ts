import { Resolver } from 'node:dns/promises'
import { request as httpsRequest } from 'node:https'
import { request as httpRequest } from 'node:http'

/**
 * Check whether a public URL actually serves, without trusting this machine's resolver.
 *
 * Observed on the development Mac: the system resolver returned nothing for
 * *.trycloudflare.com while 1.1.1.1 and 8.8.8.8 both answered instantly. A working
 * tunnel therefore looked completely dead to any check built on plain fetch(). Friends
 * elsewhere would have connected fine.
 *
 * So: try the normal path first, and on a DNS failure resolve through a public resolver
 * and connect straight to the address with the right SNI and Host header.
 */

export type ProbeResult = {
  ok: boolean
  status?: number
  body?: string
  /** How it succeeded — 'system' is the normal path; 'public-dns' means local DNS is broken. */
  via?: 'system' | 'public-dns'
  addresses?: string[]
  error?: string
  localDnsWorks: boolean
}

const PUBLIC_RESOLVERS = ['1.1.1.1', '8.8.8.8']

export async function probeUrl(url: string, timeoutMs = 8000): Promise<ProbeResult> {
  const target = new URL(url)

  // 1. The ordinary path, which is what a friend's machine will use.
  try {
    const res = await fetch(url, { signal: AbortSignal.timeout(timeoutMs) })
    const body = (await res.text()).slice(0, 2000)
    return { ok: res.ok, status: res.status, body, via: 'system', localDnsWorks: true }
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err)
    const looksLikeDns = /ENOTFOUND|EAI_AGAIN|getaddrinfo|Failed to fetch|fetch failed/i.test(message)
    if (!looksLikeDns) return { ok: false, error: message, localDnsWorks: true }
  }

  // 2. Local DNS let us down. Resolve elsewhere and connect by address.
  const resolver = new Resolver()
  resolver.setServers(PUBLIC_RESOLVERS)
  let addresses: string[] = []
  try {
    addresses = await resolver.resolve4(target.hostname)
  } catch (err) {
    return {
      ok: false,
      localDnsWorks: false,
      error: `neither this machine nor ${PUBLIC_RESOLVERS.join('/')} could resolve ${target.hostname}: ` +
        (err instanceof Error ? err.message : String(err)),
    }
  }
  if (addresses.length === 0) {
    return { ok: false, localDnsWorks: false, error: `${target.hostname} has no A record` }
  }

  const result = await new Promise<ProbeResult>(resolve => {
    const isHttps = target.protocol === 'https:'
    const req = (isHttps ? httpsRequest : httpRequest)({
      host: addresses[0],
      port: target.port ? Number(target.port) : (isHttps ? 443 : 80),
      path: target.pathname + target.search,
      method: 'GET',
      // The certificate is for the hostname, not the address we dialled.
      servername: target.hostname,
      headers: { host: target.hostname },
      timeout: timeoutMs,
    }, res => {
      let body = ''
      res.on('data', (c: Buffer) => { if (body.length < 2000) body += c.toString('utf8') })
      res.on('end', () => resolve({
        ok: (res.statusCode ?? 0) >= 200 && (res.statusCode ?? 0) < 400,
        status: res.statusCode,
        body,
        via: 'public-dns',
        addresses,
        localDnsWorks: false,
      }))
    })
    req.on('timeout', () => { req.destroy(); resolve({ ok: false, error: 'timed out', addresses, localDnsWorks: false }) })
    req.on('error', err => resolve({ ok: false, error: err.message, addresses, localDnsWorks: false }))
    req.end()
  })

  return result
}
