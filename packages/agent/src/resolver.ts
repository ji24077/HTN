import { Resolver } from 'node:dns/promises'
import { lookup as systemLookup } from 'node:dns'
import type { LookupFunction } from 'node:net'
import { Agent, setGlobalDispatcher } from 'undici'

/**
 * Keep working on a network whose DNS is unreliable.
 *
 * Observed in practice on phone tethering: the system resolver handles established
 * domains but returns nothing for a newly created subdomain, while 1.1.1.1 answers it
 * instantly. The address is fine and the server is fine — only this machine's resolver
 * disagrees, and without a fallback the agent simply cannot join.
 *
 * So: always try the system resolver first (it is authoritative for private and
 * split-horizon names, which a public resolver would get wrong), and fall back to public
 * resolvers only when it has nothing. Enable with DWP_DNS_FALLBACK=1.
 */

const PUBLIC_RESOLVERS = ['1.1.1.1', '8.8.8.8']
const cache = new Map<string, { addresses: string[]; expires: number }>()
const TTL_MS = 60_000

async function resolveViaPublic(hostname: string): Promise<string[]> {
  const hit = cache.get(hostname)
  if (hit && hit.expires > Date.now()) return hit.addresses

  const resolver = new Resolver()
  resolver.setServers(PUBLIC_RESOLVERS)
  const addresses = await resolver.resolve4(hostname)
  cache.set(hostname, { addresses, expires: Date.now() + TTL_MS })
  return addresses
}

/** A drop-in replacement for dns.lookup that falls back to public resolvers. */
export const fallbackLookup: LookupFunction = ((hostname, options, callback) => {
  const cb = (typeof options === 'function' ? options : callback) as (
    err: NodeJS.ErrnoException | null, address?: string | { address: string; family: number }[], family?: number,
  ) => void
  const opts = typeof options === 'function' ? {} : (options ?? {})

  systemLookup(hostname, opts as never, (err, address, family) => {
    if (!err) return cb(null, address as never, family)

    resolveViaPublic(hostname)
      .then(addresses => {
        if (addresses.length === 0) return cb(err)
        // Honour the caller's shape: `all: true` expects an array.
        if ((opts as { all?: boolean }).all) {
          cb(null, addresses.map(a => ({ address: a, family: 4 })) as never, 4)
        } else {
          cb(null, addresses[0] as never, 4)
        }
      })
      .catch(() => cb(err))
  })
}) as LookupFunction

/**
 * Dial options that bypass a broken resolver, for transports that ignore `lookup`.
 *
 * The lookup hook works for Node's `ws` and for fetch, but a Bun-compiled binary uses its
 * own WebSocket implementation and ignores it — so a machine whose resolver cannot see a
 * freshly created hostname could pair (fetch honoured the fallback) and then never
 * connect. Resolving here and dialling the address directly closes that gap, with the
 * certificate still checked against the real hostname via SNI.
 */
export async function directDial(wsUrl: string): Promise<null | {
  url: string
  options: { servername: string; headers: Record<string, string> }
}> {
  if (!dnsFallbackEnabled()) return null
  const url = new URL(wsUrl)

  // Only step in when the system resolver genuinely cannot answer.
  try {
    await new Promise<void>((resolve, reject) =>
      systemLookup(url.hostname, err => (err ? reject(err) : resolve())))
    return null
  } catch {}

  let addresses: string[]
  try {
    addresses = await resolveViaPublic(url.hostname)
  } catch {
    return null
  }
  if (addresses.length === 0) return null

  const port = url.port || (url.protocol === 'wss:' ? '443' : '80')
  return {
    url: `${url.protocol}//${addresses[0]}:${port}${url.pathname}${url.search}`,
    // SNI and Host stay the real name, so TLS verification is unchanged — only the
    // address we connect to is supplied by a different resolver.
    options: { servername: url.hostname, headers: { host: url.hostname } },
  }
}

export const dnsFallbackEnabled = (): boolean =>
  process.env.DWP_DNS_FALLBACK === '1' || process.env.DWP_DNS_FALLBACK === 'true'

/** Route fetch() through the fallback resolver too, not just WebSocket dials. */
export function installDnsFallback(): void {
  if (!dnsFallbackEnabled()) return
  setGlobalDispatcher(new Agent({ connect: { lookup: fallbackLookup } }))
}
