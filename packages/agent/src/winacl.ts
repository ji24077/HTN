/**
 * Windows-only protection for the host's private key.
 *
 * Nothing in this file runs on macOS or Linux — `keys.ts` calls it only under win32, and
 * the POSIX mode check it sits beside is untouched. It exists because the POSIX control
 * has no meaning here: Windows has no mode bits, only a read-only attribute, so Node
 * reports 0o666 for every writable file and `chmod 600` is not a thing a person can run.
 *
 * The equivalent control is an ACL that grants the current user and nobody else. This
 * sets one at key creation and reports what it managed to do. It is deliberately
 * best-effort: a key that cannot be locked down is worth saying out loud, but it is not
 * worth refusing to start over, because unlike a group-readable file on a shared Unix
 * box the common case here is a single-user laptop.
 */
import { execFileSync } from 'node:child_process'

export type AclResult = { ok: boolean; detail: string }

/** The current user's SID — locale-independent, unlike every display name Windows prints. */
function currentUserSid(): string | null {
  try {
    // /fo csv /nh gives exactly: "domain\user","S-1-5-21-..."
    const out = execFileSync('whoami', ['/user', '/fo', 'csv', '/nh'], { stdio: 'pipe' }).toString()
    return /(S-1-[0-9-]+)/.exec(out)?.[1] ?? null
  } catch {
    return null
  }
}

/**
 * Remove inherited access and grant full control to the current user alone.
 *
 * `*<SID>` is icacls' own syntax for naming a principal by SID, which avoids matching on
 * group names that are translated on non-English installs.
 */
export function restrictToCurrentUser(path: string): AclResult {
  const sid = currentUserSid()
  if (!sid) return { ok: false, detail: 'could not determine this account SID (whoami failed)' }

  try {
    execFileSync('icacls', [path, '/inheritance:r', '/grant:r', `*${sid}:F`], { stdio: 'pipe' })
    return { ok: true, detail: 'ACL restricted to this account' }
  } catch (err) {
    return { ok: false, detail: err instanceof Error ? err.message.split('\n')[0]! : String(err) }
  }
}
