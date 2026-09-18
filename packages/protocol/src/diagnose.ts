import { Resolver } from 'node:dns/promises'
import { lookup } from 'node:dns/promises'

/**
 * Work out why an address could not be reached, and say it in plain language.
 *
 * "fetch failed" and a stack trace are useless to someone trying to join a friend's
 * network. Each branch below distinguishes a cause the person can actually act on.
 */

export type Diagnosis = {
  reachable: boolean
  /** Short machine-readable cause, for logs. */
  cause:
    | 'ok'
    | 'dns-local-only'
    | 'dns-nowhere'
    | 'refused'
    | 'timeout'
    | 'tls'
    | 'not-a-server'
    | 'unknown'
  /** What to tell the person, already wrapped and indented. */
  message: string
  detail?: string
}

const PUBLIC_RESOLVERS = ['1.1.1.1', '8.8.8.8']

export async function diagnoseOrigin(origin: string): Promise<Diagnosis> {
  let url: URL
  try {
    url = new URL(origin)
  } catch {
    return {
      reachable: false,
      cause: 'unknown',
      message: `"${origin}" is not a valid address.\n  It should look like  https://something.example.com`,
    }
  }
  const host = url.hostname

  // Does this machine's own resolver know the name?
  let systemResolved = true
  try {
    await lookup(host)
  } catch {
    systemResolved = false
  }

  if (!systemResolved) {
    let publicAddresses: string[] = []
    try {
      const resolver = new Resolver()
      resolver.setServers(PUBLIC_RESOLVERS)
      publicAddresses = await resolver.resolve4(host)
    } catch {
      publicAddresses = []
    }

    if (publicAddresses.length > 0) {
      return {
        reachable: false,
        cause: 'dns-local-only',
        detail: `resolved to ${publicAddresses.join(', ')} via ${PUBLIC_RESOLVERS[0]}`,
        message:
          `This computer cannot look up "${host}", but public DNS servers can.\n` +
          `  The address itself is fine — the problem is this machine's DNS.\n\n` +
          `  This is common on phone tethering and some home routers, which are slow to\n` +
          `  pick up newly created names.\n\n` +
          `  Try one of:\n` +
          `    - set this computer's DNS to 1.1.1.1 or 8.8.8.8\n` +
          `    - switch to a different network (or off tethering)\n` +
          `    - ask for an address on a domain that already exists, rather than a\n` +
          `      freshly created one`,
      }
    }

    return {
      reachable: false,
      cause: 'dns-nowhere',
      message:
        `No DNS server anywhere knows the name "${host}".\n\n` +
        `  Either it is mistyped, or the address has expired. Temporary tunnel addresses\n` +
        `  change every time the host restarts — ask them for the current link.`,
    }
  }

  // The name resolves, so try to talk to it.
  try {
    const res = await fetch(`${url.origin}/health`, { signal: AbortSignal.timeout(10_000) })
    if (res.ok) {
      const body = await res.json().catch(() => null) as { ok?: boolean } | null
      if (body?.ok) return { reachable: true, cause: 'ok', message: 'The address is reachable.' }
      return {
        reachable: false,
        cause: 'not-a-server',
        message: `Something answered at ${url.origin}, but it is not this system's server.\n  Check the address is the one you were given.`,
      }
    }
    return {
      reachable: false,
      cause: 'not-a-server',
      detail: `HTTP ${res.status}`,
      message:
        `${url.origin} answered with HTTP ${res.status}.\n\n` +
        (res.status === 502 || res.status === 503 || res.status === 530
          ? `  That usually means the tunnel is up but the server behind it is not running.\n` +
            `  Ask the host to check their server is started.`
          : `  Check the address is correct.`),
    }
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err)
    if (/timed? ?out|ETIMEDOUT|AbortError/i.test(message)) {
      return {
        reachable: false,
        cause: 'timeout',
        detail: message,
        message:
          `${url.origin} did not answer in time.\n\n` +
          `  The host's computer may be asleep or offline, or this network may be\n` +
          `  blocking the connection.`,
      }
    }
    if (/ECONNREFUSED/i.test(message)) {
      return {
        reachable: false,
        cause: 'refused',
        detail: message,
        message: `${url.origin} refused the connection.\n\n  Nothing is listening there. Ask the host whether their server is running.`,
      }
    }
    if (/certificate|TLS|SSL|altname/i.test(message)) {
      return {
        reachable: false,
        cause: 'tls',
        detail: message,
        message:
          `The secure connection to ${url.origin} could not be established.\n\n` +
          `  ${message}\n\n` +
          `  Some workplace networks inspect encrypted traffic and break it this way.`,
      }
    }
    return { reachable: false, cause: 'unknown', detail: message, message: `Could not reach ${url.origin}.\n\n  ${message}` }
  }
}
