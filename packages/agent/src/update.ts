import { execFile } from 'node:child_process'
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { promisify } from 'node:util'
import type { KeyObject } from 'node:crypto'
import { SignedRelease, verifyRelease, hashBytes, mintAssertion, createLogger } from '@dwp/protocol'
import { loadConfig, saveConfig, type AgentConfig } from './config.ts'

const exec = promisify(execFile)
const log = createLogger({ component: 'agent' })

/** Where this agent is installed — three levels up from packages/agent/src. */
export function installRoot(): string {
  return join(dirname(fileURLToPath(import.meta.url)), '..', '..', '..')
}

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
  opts: { force?: boolean } = {},
): Promise<UpdateResult> {
  if (!cfg.releaseKey) {
    return { status: 'refused', reason: 'this agent pinned no release key when it paired, so it cannot verify updates' }
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

  if (!verifyRelease(signed, cfg.releaseKey)) {
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
    await exec('tar', ['-xzf', archive, '-C', root])
    await exec('pnpm', ['install', '--silent'], { cwd: root, timeout: 600_000 })

    const previous = cfg.installedRelease ?? null
    saveConfig({ ...cfg, installedRelease: signed.manifest.version })
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
  const { spawn } = require('node:child_process') as typeof import('node:child_process')
  const child = spawn(process.execPath, process.argv.slice(1), {
    detached: true,
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
