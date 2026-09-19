import { execFile, spawn } from 'node:child_process'
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { promisify } from 'node:util'
import type { KeyObject } from 'node:crypto'
import { SignedRelease, verifyRelease, hashBytes, mintAssertion, createLogger } from '@dwp/protocol'
import { loadConfig, saveConfig, type AgentConfig } from './config.ts'
import { installRoot } from './paths.ts'
import { installWorkload } from './workloads.ts'

const exec = promisify(execFile)
const log = createLogger({ component: 'agent' })

const IS_WINDOWS = process.platform === 'win32'

export type UpdateResult =
  | { status: 'current'; version: string }
  | { status: 'updated'; from: string | null; to: string }
  | { status: 'unavailable'; reason: string }
  | { status: 'refused'; reason: string }

/**
 * Fetch, verify and install the release the server is offering.
 *
 * The verification order is deliberate: signature first, then hash, then unpack. A
 * bundle that fails either check is never written anywhere it could be executed, and
 * the signature is checked against the key pinned at pairing rather than the one the
 * release arrives with — otherwise the signature would be decorative.
 */
export async function applyUpdate(
  cfg: AgentConfig,
  privateKey: KeyObject,
  opts: { force?: boolean; trustOnFirstUse?: boolean } = {},
): Promise<UpdateResult> {
  // An agent that paired before this server offered signed releases has nothing to
  // verify against. It can adopt the key it is offered, but only on a deliberate,
  // explicit instruction — never silently, and never as part of an automatic update.
  if (!cfg.releaseKey && !opts.trustOnFirstUse) {
    return {
      status: 'refused',
      reason: 'this computer pinned no release key when it paired, so it cannot verify updates.\n' +
        '  Run  pnpm agent trust-updates  to see the key and decide, or pair again to pin it automatically',
    }
  }

  let signed: SignedRelease
  try {
    const res = await fetch(`${cfg.server}/release/latest`, { signal: AbortSignal.timeout(20_000) })
    if (res.status === 404) return { status: 'unavailable', reason: 'the server is not offering a release' }
    if (!res.ok) return { status: 'unavailable', reason: `HTTP ${res.status}` }
    signed = SignedRelease.parse(await res.json())
  } catch (err) {
    return { status: 'unavailable', reason: err instanceof Error ? err.message : String(err) }
  }

  const pinned = cfg.releaseKey ?? signed.publicKey
  if (!verifyRelease(signed, pinned)) {
    // Either the server is serving something the operator did not sign, or the operator
    // rotated keys. Both need a human; neither justifies running the code.
    log.error('update.signature_rejected', { version: signed.manifest.version })
    return {
      status: 'refused',
      reason: 'the release was not signed by the key this computer pinned when it paired — refusing to install it',
    }
  }

  if (!opts.force && cfg.installedRelease === signed.manifest.version) {
    return { status: 'current', version: signed.manifest.version }
  }

  const res = await fetch(`${cfg.server}/release/${signed.manifest.sha256}`, {
    headers: { authorization: `Bearer ${mintAssertion(cfg.hostId, privateKey)}` },
    signal: AbortSignal.timeout(180_000),
  })
  if (!res.ok) return { status: 'unavailable', reason: `downloading the bundle failed: HTTP ${res.status}` }

  const bytes = Buffer.from(await res.arrayBuffer())
  if (hashBytes(bytes) !== signed.manifest.sha256) {
    return { status: 'refused', reason: 'the downloaded bundle does not match the hash that was signed' }
  }

  const staging = mkdtempSync(join(tmpdir(), 'dwp-update-'))
  try {
    const archive = join(staging, 'release.tar.gz')
    writeFileSync(archive, bytes)
    const root = installRoot()

    // Unpack over the install, then reconcile dependencies. Only source is replaced;
    // ~/.dwp — keys, config, cached artifacts — is a separate directory and untouched.
    //
    // tar is bsdtar on Windows 10 1803 and newer, so the same invocation works there.
    // pnpm is not: it is a .cmd shim, which Node refuses to spawn without a shell, so a
    // Windows host would report every update as unavailable with an opaque EINVAL.
    await exec('tar', ['-xzf', archive, '-C', root])
    await exec('pnpm', ['install', '--silent'], { cwd: root, timeout: 600_000, shell: IS_WINDOWS })

    const previous = cfg.installedRelease ?? null
    saveConfig({ ...cfg, installedRelease: signed.manifest.version, releaseKey: pinned })
    log.info('update.installed', { from: previous, to: signed.manifest.version })
    return { status: 'updated', from: previous, to: signed.manifest.version }
  } catch (err) {
    return { status: 'unavailable', reason: err instanceof Error ? err.message : String(err) }
  } finally {
    rmSync(staging, { recursive: true, force: true })
  }
}

/**
 * Restart into the freshly installed code.
 *
 * Detached, so the new process outlives this one rather than dying with it, and with
 * stdio inherited so whoever is watching the window keeps seeing output.
 */
export function restartIntoNewVersion(): never {
  // `require` does not exist in an ES module. An earlier version called it here, so the
  // update installed correctly and then the restart threw — leaving the agent updated
  // and dead, which is the worst of both outcomes and invisible to whoever owns the
  // machine. Import it properly instead.
  //
  // `detached` is POSIX-only on purpose. There it is what lets the replacement escape
  // this process group and survive; on Windows a child already outlives its parent, and
  // detaching additionally means DETACHED_PROCESS — no console — so an agent restarted
  // that way would keep running with its output going nowhere, in the one window its
  // owner is watching.
  const child = spawn(process.execPath, process.argv.slice(1), {
    detached: !IS_WINDOWS,
    stdio: 'inherit',
    cwd: process.cwd(),
    env: process.env,
  })
  child.unref()
  process.exit(0)
}

export function currentVersion(): string | null {
  const cfg = loadConfig()
  if (cfg?.installedRelease) return cfg.installedRelease
  try {
    return (JSON.parse(readFileSync(join(installRoot(), 'packages/agent/package.json'), 'utf8')) as { version: string }).version
  } catch {
    return null
  }
}
